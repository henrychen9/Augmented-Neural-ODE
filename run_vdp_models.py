#run_vdp_models.py 
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import data
from configs import problem_configs
from model import PolynomialODE
from baseline_models import PolyLinearODE
from train import train_pnode


# ----------------------------
# config
# ----------------------------

problem_fn = data.van_der_pol
cfg = problem_configs[problem_fn]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# force degrees
pnode_degree = cfg.get("degree", 2)   # keep your pnode degree unless you want to force 2
baseline_degree = 2                   # baseline fixed to quadratic

epochs = 1000
out_dir = "viz_vdp"
os.makedirs(out_dir, exist_ok=True)

# grid for heatmaps in [-1, 1]^2
grid_lim = 1.0
grid_n = 81  # 81x81 looks good


# ----------------------------
# data reconstruction (same idea as your sweep/main)
# ----------------------------

t0, t1, N = cfg["t0"], cfg["t1"], cfg["N"]
input_dim = cfg["input_dim"]
output_dim = cfg["output_dim"]
method = cfg["method"]
lr = cfg["lr"]
noise_level = cfg["noise_level"]
to_normalize = cfg["to_normalize"]

assert input_dim == 2, "this visualization script is written for 2d systems"

y0s = torch.stack(cfg["initial_conditions"]).to(device)
t_train = torch.linspace(t0, t1, N, device=device)

true_ys = []
for y0 in y0s:
    _, y = data.gen_trajectory(
        t0=t0, t1=t1, steps=N,
        y0=y0, device=device,
        problem=problem_fn
    )
    true_ys.append(y)
true_ys = torch.stack(true_ys, dim=0)  # [B,T,2]

scale = None
if to_normalize:
    scale = true_ys.abs().amax(dim=(0, 1)).clamp(min=1e-8)
    true_ys = true_ys / scale
    y0s = y0s / scale

with torch.no_grad():
    std = noise_level * true_ys.abs().mean()
    true_ys_noisy = true_ys + std * torch.randn_like(true_ys)


# ----------------------------
# grid eval helpers ([-1,1]^2)
# ----------------------------

@torch.no_grad()
def eval_vector_field_on_grid(model, true_problem, grid_lim=1.0, grid_n=81, device="cpu"):
    # build grid points
    xs = torch.linspace(-grid_lim, grid_lim, grid_n, device=device)
    ys = torch.linspace(-grid_lim, grid_lim, grid_n, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)  # [N,2]

    # learned field (batch ok)
    f_learn = model(None, pts)  # [N,2]

    # true field (your true fns are not batch-safe, so we loop)
    f_true_list = []
    pts_cpu = pts.detach().cpu()
    for i in range(pts_cpu.shape[0]):
        p = pts_cpu[i]
        f_true_list.append(true_problem(None, p))
    f_true = torch.stack(f_true_list, dim=0).to(device)

    # reshape back to grid
    fL = f_learn.reshape(grid_n, grid_n, 2)
    fT = f_true.reshape(grid_n, grid_n, 2)

    # magnitudes for heatmap
    mag_L = torch.linalg.norm(fL, dim=2)
    mag_T = torch.linalg.norm(fT, dim=2)
    mag_E = torch.linalg.norm(fL - fT, dim=2)

    return Xg, Yg, mag_T, mag_L, mag_E


def save_heatmap(mat, title, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(
        mat,
        origin="lower",
        extent=[-grid_lim, grid_lim, -grid_lim, grid_lim],
        aspect="equal"
    )
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("v")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ----------------------------
# define normalized true problem if needed
# ----------------------------

if to_normalize and scale is not None:
    # your model is trained in normalized coordinates, so true vector field must match that space
    # y_norm = y / scale  => dy_norm/dt = f(y_norm * scale) / scale
    true_problem = (lambda t, y: problem_fn(t, y * scale.cpu()) / scale.cpu())
else:
    true_problem = problem_fn


# ----------------------------
# train pnode
# ----------------------------

print("\ntraining pnode...")
pnode = PolynomialODE(
    input_dim=input_dim,
    hidden_dims=[40, 40],
    degree=pnode_degree,
    output_dim=output_dim
).to(device)

pnode, pnode_loss_hist, _ = train_pnode(
    pnode,
    y0s,
    t_train,
    true_ys_noisy,
    method=method,
    device=device,
    lr=lr,
    epochs=epochs
)

print("pnode final loss:", float(pnode_loss_hist[-1]))


# ----------------------------
# train baseline (quadratic)
# ----------------------------

print("\ntraining polylinear baseline...")
baseline = PolyLinearODE(
    input_dim=input_dim,
    degree=baseline_degree,
    output_dim=output_dim
).to(device)

baseline, base_loss_hist, _ = train_pnode(
    baseline,
    y0s,
    t_train,
    true_ys_noisy,
    method=method,
    device=device,
    lr=lr,
    epochs=epochs
)

print("baseline final loss:", float(base_loss_hist[-1]))


# ----------------------------
# heatmap visualization inside [-1,1]^2 for both
# ----------------------------

print("\ncomputing grid heatmaps...")

# pnode
_, _, magT, magL, magE = eval_vector_field_on_grid(
    pnode, true_problem, grid_lim=grid_lim, grid_n=grid_n, device=device
)
save_heatmap(magT.detach().cpu().numpy(), "true |f(x)| (vdp)", os.path.join(out_dir, "true_mag.png"))
save_heatmap(magL.detach().cpu().numpy(), "pnode learned |f(x)|", os.path.join(out_dir, "pnode_mag.png"))
save_heatmap(magE.detach().cpu().numpy(), "pnode error |f(x)-f_true(x)|", os.path.join(out_dir, "pnode_err.png"))

# baseline
_, _, magT2, magL2, magE2 = eval_vector_field_on_grid(
    baseline, true_problem, grid_lim=grid_lim, grid_n=grid_n, device=device
)
save_heatmap(magL2.detach().cpu().numpy(), "baseline learned |f(x)| (degree=2)", os.path.join(out_dir, "baseline_mag.png"))
save_heatmap(magE2.detach().cpu().numpy(), "baseline error |f(x)-f_true(x)|", os.path.join(out_dir, "baseline_err.png"))

print("\ndone. saved heatmaps to:", out_dir)
print("  true_mag.png")
print("  pnode_mag.png, pnode_err.png")
print("  baseline_mag.png, baseline_err.png")
