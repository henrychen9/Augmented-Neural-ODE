# sweep_inits.py
# run many random initializations, label runs as accurate vs non-accurate,
# and log weight/activation/feature stats to diagnose instability

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torchdiffeq import odeint

import data
from model import PolynomialODE
from train import train_pnode
from configs import problem_configs


# ----------------------------
# utilities
# ----------------------------

def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # may slow things down and can raise errors for some ops
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rel_mse(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # mean( ||a-b||^2 / (||b||^2 + eps) ) over batch
    num = ((a - b) ** 2).sum(dim=-1)
    den = (b ** 2).sum(dim=-1).clamp(min=eps)
    return (num / den).mean()


def flatten_dict(d, parent_key="", sep="."):
    out = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, new_key, sep=sep))
        else:
            out[new_key] = v
    return out


# ----------------------------
# stats extraction
# ----------------------------

@torch.no_grad()
def compute_feature_stats(model: PolynomialODE, points: torch.Tensor) -> dict:
    # points: [N, input_dim]
    phi = model.feature_transform(points)  # [N, poly_dim]
    stats = {
        "phi_mean": phi.mean().item(),
        "phi_std": phi.std(unbiased=False).item(),
        "phi_abs_mean": phi.abs().mean().item(),
        "phi_abs_max": phi.abs().max().item(),
    }
    return stats


@torch.no_grad()
def compute_activation_stats(model: PolynomialODE, z: torch.Tensor) -> dict:
    """
    Collect pre-activation stats for each Linear layer in model.net on input features z.
    z: [N, poly_dim]
    """
    stats = {}
    x = z

    layer_idx = 0
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            pre = layer(x)  # pre-activation
            stats[f"linear_{layer_idx}.pre_mean"] = pre.mean().item()
            stats[f"linear_{layer_idx}.pre_std"] = pre.std(unbiased=False).item()
            stats[f"linear_{layer_idx}.pre_abs_mean"] = pre.abs().mean().item()
            stats[f"linear_{layer_idx}.pre_abs_max"] = pre.abs().max().item()
            # "saturation-ish" heuristic: big |pre| means GELU is close to linear but can still create scale issues
            stats[f"linear_{layer_idx}.pre_frac_abs_gt6"] = (pre.abs() > 6.0).float().mean().item()
            x = pre
            layer_idx += 1
        else:
            x = layer(x)

    return stats


@torch.no_grad()
def compute_weight_stats(model: PolynomialODE) -> dict:
    """
    Focus on:
    - first Linear layer W1 and b1
    - last Linear layer W_last and b_last
    - overall parameter norms
    """
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    if len(linears) < 2:
        raise RuntimeError("expected at least 2 Linear layers in model.net")

    first = linears[0]
    last = linears[-1]

    def frob_norm(W):
        return torch.linalg.norm(W, ord="fro").item()

    def vec_norm(v):
        return torch.linalg.norm(v).item()

    stats = {}

    # first layer
    W1 = first.weight.detach()
    stats["W1_frob"] = frob_norm(W1)
    stats["W1_row_norm_mean"] = torch.linalg.norm(W1, dim=1).mean().item()
    stats["W1_row_norm_std"] = torch.linalg.norm(W1, dim=1).std(unbiased=False).item()
    if first.bias is not None:
        b1 = first.bias.detach()
        stats["b1_norm"] = vec_norm(b1)
        stats["b1_abs_mean"] = b1.abs().mean().item()
        stats["b1_abs_max"] = b1.abs().max().item()

    # singular values for conditioning
    # for speed, compute svdvals on CPU if needed
    try:
        s = torch.linalg.svdvals(W1)
        s_max = s.max().item()
        s_min = s.min().item()
        stats["W1_smax"] = s_max
        stats["W1_smin"] = s_min
        stats["W1_cond"] = (s_max / max(s_min, 1e-12))
        # store a couple more summaries
        stats["W1_smean"] = s.mean().item()
    except RuntimeError:
        # some CUDA / driver combos can fail on svdvals
        stats["W1_smax"] = float("nan")
        stats["W1_smin"] = float("nan")
        stats["W1_cond"] = float("nan")
        stats["W1_smean"] = float("nan")

    # last layer
    Wl = last.weight.detach()
    stats["Wlast_frob"] = frob_norm(Wl)
    stats["Wlast_row_norm_mean"] = torch.linalg.norm(Wl, dim=1).mean().item()
    stats["Wlast_row_norm_std"] = torch.linalg.norm(Wl, dim=1).std(unbiased=False).item()
    if last.bias is not None:
        bl = last.bias.detach()
        stats["blast_norm"] = vec_norm(bl)
        stats["blast_abs_mean"] = bl.abs().mean().item()
        stats["blast_abs_max"] = bl.abs().max().item()

    # overall parameter norm
    total_sq = 0.0
    for p in model.parameters():
        total_sq += float((p.detach() ** 2).sum().item())
    stats["param_l2"] = math.sqrt(total_sq)

    return stats


# ----------------------------
# evaluation metrics
# ----------------------------

@torch.no_grad()
def eval_heldout_traj_mse(
    model: PolynomialODE,
    problem_fn,
    t0: float,
    t1: float,
    steps: int,
    y0s_eval: torch.Tensor,
    device: torch.device,
    method: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict:
    """
    Compare predicted trajectories vs true trajectories on held-out ICs.
    Returns absolute MSE and relative MSE.
    """
    t = torch.linspace(t0, t1, steps, device=device)

    # true
    true = []
    for y0 in y0s_eval:
        _, y = data.gen_trajectory(t0=t0, t1=t1, steps=steps, y0=y0, device=device, problem=problem_fn)
        true.append(y)
    true = torch.stack(true, dim=0)  # [B, T, d]

    # pred
    pred = odeint(model, y0s_eval, t, rtol=rtol, atol=atol, method=method)  # [T, B, d]
    pred = pred.permute(1, 0, 2).contiguous()  # [B, T, d]

    mse = torch.mean((pred - true) ** 2).item()
    rel = rel_mse(pred.view(-1, pred.shape[-1]), true.view(-1, true.shape[-1])).item()

    return {"heldout_mse": mse, "heldout_rel_mse": rel}


@torch.no_grad()
def eval_vectorfield_grid(
    model: PolynomialODE,
    problem_fn,
    input_dim: int,
    device: torch.device,
    grid_lim: float = 1.0,
    grid_n: int = 25,
) -> dict:
    """
    Sample a grid in [-grid_lim, grid_lim]^d and compare f_learned vs f_true.
    Returns absolute MSE and relative MSE (per-point normalized).
    """
    axes = [torch.linspace(-grid_lim, grid_lim, grid_n, device=device) for _ in range(input_dim)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    pts = torch.stack([m.reshape(-1) for m in mesh], dim=1)  # [N, d]

    f_learn = model(None, pts)  # [N, d]
    f_true = torch.stack([problem_fn(None, p) for p in pts.detach().cpu()]).to(device)

    mse = torch.mean((f_learn - f_true) ** 2).item()
    rel = rel_mse(f_learn, f_true).item()

    # log size of the vector field too
    f_learn_abs_max = f_learn.abs().max().item()
    f_true_abs_max = f_true.abs().max().item()

    return {
        "vf_grid_mse": mse,
        "vf_grid_rel_mse": rel,
        "vf_learn_abs_max": f_learn_abs_max,
        "vf_true_abs_max": f_true_abs_max,
        "vf_grid_npts": int(pts.shape[0]),
    }


@torch.no_grad()
def extract_initial_weights(model: PolynomialODE) -> dict:
    """
    Extract initial weights and biases for ALL Linear layers in model.net.
    Stored in order: linear_0, linear_1, ..., linear_k
    """
    out = {}

    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]

    for idx, layer in enumerate(linears):
        out[f"linear_{idx}.W"] = layer.weight.detach().cpu()
        if layer.bias is not None:
            out[f"linear_{idx}.b"] = layer.bias.detach().cpu()

    return out



# ----------------------------
# main sweep
# ----------------------------

@dataclass
class RunRecord:
    seed: int
    train_loss_final: float
    nfe_final: int
    epoch_stop: int
    heldout_mse: float
    heldout_rel_mse: float
    vf_grid_mse: float
    vf_grid_rel_mse: float
    label: str
    stats: dict
    init_weights: dict   # NEW


def make_heldout_ics(
    n: int,
    input_dim: int,
    lo: float,
    hi: float,
    device: torch.device,
) -> torch.Tensor:
    y0 = (hi - lo) * torch.rand(n, input_dim, device=device) + lo
    return y0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="quartic_duffing",
                        help="one of: quartic_duffing, duffing, van_der_pol, soft_nonlinear, pendulum, sine_oscillator, lorenz, lotka_volterra, pitchfork")
    parser.add_argument("--degree", type=int, default=2, help="force polynomial degree for the model (and feature library)")
    parser.add_argument("--runs", type=int, default=200, help="number of random initializations")
    parser.add_argument("--seed0", type=int, default=0, help="starting seed")
    parser.add_argument("--epochs", type=int, default=1000, help="max epochs per run")
    parser.add_argument("--deterministic", action="store_true", help="use deterministic torch settings")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda")
    parser.add_argument("--heldout_B", type=int, default=12, help="held-out IC count")
    parser.add_argument("--heldout_lo", type=float, default=-1.5, help="held-out IC lower bound")
    parser.add_argument("--heldout_hi", type=float, default=1.5, help="held-out IC upper bound")
    parser.add_argument("--heldout_steps", type=int, default=600, help="time steps for held-out evaluation")
    parser.add_argument("--grid_lim", type=float, default=1.0, help="vector field grid limit")
    parser.add_argument("--grid_n", type=int, default=25, help="vector field grid resolution per axis")
    parser.add_argument("--acc_thresh", type=float, default=1e-3, help="threshold on heldout_mse for accurate label")
    parser.add_argument("--out", type=str, default="sweep_degree2.pt", help="output .pt file to save results")
    args = parser.parse_args()

    # map name -> function object
    name_to_problem = {
        "linear": data.linear,
        "damped": data.damped,
        "van_der_pol": data.van_der_pol,
        "rayleigh": data.rayleigh,
        "pitchfork": data.pitchfork,
        "lorenz": data.lorenz,
        "pendulum": data.pendulum,
        "sine_oscillator": data.sine_oscillator,
        "duffing": data.duffing,
        "lotka_volterra": data.lotka_volterra,
        "soft_nonlinear": data.soft_nonlinear,
        "quartic_duffing": data.quartic_duffing,
    }
    if args.problem not in name_to_problem:
        raise ValueError(f"unknown problem '{args.problem}'")

    problem_fn = name_to_problem[args.problem]
    cfg = problem_configs[problem_fn]

    # force degree in model regardless of configs
    degree = int(args.degree)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # pull config values
    t0, t1, N = cfg["t0"], cfg["t1"], cfg["N"]
    input_dim, output_dim = cfg["input_dim"], cfg["output_dim"]
    method = cfg["method"]
    noise_level = cfg["noise_level"]
    lr = cfg["lr"]
    to_normalize = cfg["to_normalize"]
    initial_conditions = cfg["initial_conditions"]

    # prepare training data once (same across runs)
    y0s = torch.stack(initial_conditions).to(device)
    t_train = torch.linspace(t0, t1, N, device=device)

    true_ys = []
    for y0 in y0s:
        _, y = data.gen_trajectory(t0=t0, t1=t1, steps=t_train.shape[0], y0=y0, device=device, problem=problem_fn)
        true_ys.append(y)
    true_ys = torch.stack(true_ys, dim=0)  # [B, T, d]

    # normalize (same as your main)
    scale = None
    if to_normalize:
        scale = true_ys.abs().amax(dim=(0, 1)).clamp(min=1e-8)
        true_ys = true_ys / scale
        y0s = y0s / scale

    # add noise once (so all runs see the same noisy training data)
    with torch.no_grad():
        std = noise_level * true_ys.abs().mean()
        noise = std * torch.randn_like(true_ys)
        true_ys_noisy = true_ys + noise

    # held-out ICs fixed for all runs (important)
    y0s_eval = make_heldout_ics(
        n=args.heldout_B,
        input_dim=input_dim,
        lo=args.heldout_lo,
        hi=args.heldout_hi,
        device=device,
    )
    if to_normalize and scale is not None:
        y0s_eval = y0s_eval / scale  # keep eval in same normalized coordinates

    # points for feature/activation stats (fixed)
    n_stat_points = 5000
    stat_points = (torch.rand(n_stat_points, input_dim, device=device) * 2.0) - 1.0  # [-1,1]^d
    if to_normalize and scale is not None:
        stat_points = stat_points / scale

    records = []
    t_start = time.time()

    good = 0
    bad = 0

    print(f"problem: {args.problem}")
    print(f"device: {device}")
    print(f"runs: {args.runs}, degree forced to: {degree}")
    print(f"train B: {int(y0s.shape[0])}, train steps: {int(t_train.shape[0])}")
    print(f"heldout B: {args.heldout_B}, heldout steps: {args.heldout_steps}")
    print(f"accuracy threshold (heldout_mse): {args.acc_thresh}")
    print()

    def checkpoint_save(tag: str = "") -> None:
        # build a partial save object from what we have so far
        obj = {
            "meta": {
                "problem": args.problem,
                "degree": degree,
                "runs": args.runs,
                "seed0": args.seed0,
                "epochs": args.epochs,
                "device": str(device),
                "acc_thresh": args.acc_thresh,
                "heldout_B": args.heldout_B,
                "heldout_steps": args.heldout_steps,
                "grid_lim": args.grid_lim,
                "grid_n": args.grid_n,
            },
            "records": [asdict(r) for r in records],
            "summary": {
                "accurate": good,
                "non_accurate_or_failed": bad,
                "accurate_frac": (good / max(len(records), 1)),
                "completed_runs": len(records),
            },
        }

        tmp_path = args.out + ".tmp"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, args.out)  # atomic on mac/linux

        suffix = f" {tag}" if tag else ""
        print(f"[checkpoint] saved -> {args.out} (after {len(records)} runs){suffix}", flush=True)

    for i in range(args.runs):
        seed = args.seed0 + i
        set_all_seeds(seed, deterministic=args.deterministic)

        model = PolynomialODE(
            input_dim=input_dim,
            hidden_dims=[40, 40],
            degree=degree,
            output_dim=output_dim
        ).to(device)

        
        init_weights = extract_initial_weights(model)

        try:
            model, loss_history, nfe_history = train_pnode(
                model,
                y0s,
                t_train,
                true_ys_noisy,
                method=method,
                device=device,
                lr=lr,
                epochs=args.epochs
            )
            train_loss_final = float(loss_history[-1]) if len(loss_history) else float("nan")
            nfe_final = int(nfe_history[-1]) if len(nfe_history) else int(model.get_nfe())
            epoch_stop = int(len(loss_history))

            held = eval_heldout_traj_mse(
                model=model,
                problem_fn=problem_fn if not to_normalize else (lambda t, y: problem_fn(t, y * scale) / scale),
                t0=t0,
                t1=t1,
                steps=args.heldout_steps,
                y0s_eval=y0s_eval,
                device=device,
                method=method,
            )

            vf = eval_vectorfield_grid(
                model=model,
                problem_fn=problem_fn if not to_normalize else (lambda t, y: problem_fn(t, y * scale) / scale),
                input_dim=input_dim,
                device=device,
                grid_lim=args.grid_lim,
                grid_n=args.grid_n
            )

            feat_stats = compute_feature_stats(model, stat_points)
            z = model.feature_transform(stat_points)
            act_stats = compute_activation_stats(model, z)
            w_stats = compute_weight_stats(model)

            stats = {}
            stats.update(feat_stats)
            stats.update(act_stats)
            stats.update(w_stats)

            label = "accurate" if held["heldout_mse"] < args.acc_thresh else "non_accurate"
            if label == "accurate":
                good += 1
            else:
                bad += 1

            rec = RunRecord(
                seed=seed,
                train_loss_final=train_loss_final,
                nfe_final=nfe_final,
                epoch_stop=epoch_stop,
                heldout_mse=float(held["heldout_mse"]),
                heldout_rel_mse=float(held["heldout_rel_mse"]),
                vf_grid_mse=float(vf["vf_grid_mse"]),
                vf_grid_rel_mse=float(vf["vf_grid_rel_mse"]),
                label=label,
                stats=stats,
                init_weights=init_weights
            )

            records.append(rec)

            # print EVERY run summary (so nothing is "missing")
            elapsed = time.time() - t_start
            print(
                f"[{i+1:4d}/{args.runs}] seed={seed:4d} "
                f"label={label:12s} "
                f"held_mse={rec.heldout_mse:.3e} "
                f"vf_rel={rec.vf_grid_rel_mse:.3e} "
                f"loss={rec.train_loss_final:.3e} "
                f"epochs={rec.epoch_stop:4d} "
                f"good={good} bad={bad} "
                f"elapsed={elapsed:.1f}s",
                flush=True
            )

            # checkpoint every 5 runs (and also after the first run)
            if (i + 1) == 1 or (i + 1) % 5 == 0:
                checkpoint_save()

        except (AssertionError, RuntimeError) as e:
            bad += 1
            rec = RunRecord(
                seed=seed,
                train_loss_final=float("nan"),
                nfe_final=-1,
                epoch_stop=0,
                heldout_mse=float("inf"),
                heldout_rel_mse=float("inf"),
                vf_grid_mse=float("inf"),
                vf_grid_rel_mse=float("inf"),
                label="failed",
                stats={"error": str(e)},
                init_weights=init_weights
            )

            records.append(rec)

            elapsed = time.time() - t_start
            print(
                f"[{i+1:4d}/{args.runs}] seed={seed:4d} label=failed "
                f"error={str(e)[:120]} "
                f"good={good} bad={bad} elapsed={elapsed:.1f}s",
                flush=True
            )

            # checkpoint failures too so you never lose progress
            if (i + 1) == 1 or (i + 1) % 5 == 0:
                checkpoint_save(tag="(after failure)")

    # save
    out_obj = {
        "meta": {
            "problem": args.problem,
            "degree": degree,
            "runs": args.runs,
            "seed0": args.seed0,
            "epochs": args.epochs,
            "device": str(device),
            "acc_thresh": args.acc_thresh,
            "heldout_B": args.heldout_B,
            "heldout_steps": args.heldout_steps,
            "grid_lim": args.grid_lim,
            "grid_n": args.grid_n,
            "config_used": {k: (str(v) if callable(v) else v) for k, v in cfg.items()},
        },
        "records": [asdict(r) for r in records],
        "summary": {
            "accurate": good,
            "non_accurate_or_failed": bad,
            "accurate_frac": (good / max(args.runs, 1)),
        }
    }

    torch.save(out_obj, args.out)

    print()
    print("done.")
    print(f"saved: {args.out}")
    print(f"accurate: {good}, non-accurate/failed: {bad}, frac: {good / max(args.runs, 1):.3f}")


if __name__ == "__main__":
    main()
