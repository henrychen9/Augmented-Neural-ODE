
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

import data
from configs import problem_configs
from model import PolynomialFeatures


class SimplePolynomialODE(nn.Module):
    # baseline: q(x; p) = W Phi(x)
    # for 1d, degree=2: Phi(x) = [1, x, x^2], so q(x; p) = p0 + p1 x + p2 x^2
    # for d-dimensional systems, Phi(x) contains all monomials up to the chosen degree

    def __init__(self, input_dim, output_dim=None, degree=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.degree = degree

        self.poly = PolynomialFeatures(input_dim=input_dim, degree=degree)
        n_features = self.poly.exponents.shape[0]

        self.linear = nn.Linear(n_features, self.output_dim, bias=False)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1

        if x.dim() == 1:
            x = x.unsqueeze(0)
            was_1d = True
        else:
            was_1d = False

        Phi = self.poly(x)
        out = self.linear(Phi)

        if was_1d:
            out = out.squeeze(0)

        return out

    def reset_nfe(self):
        self.nfe = 0

    def get_nfe(self):
        return self.nfe

    def get_equations(self, var_names=None, tol=1e-6):
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.input_dim)]

        weights = self.linear.weight.detach().cpu().numpy()
        exps = self.poly.exponents.detach().cpu().numpy().astype(int)

        eqs = []
        for i in range(self.output_dim):
            terms = []
            for c, exp in zip(weights[i], exps):
                if abs(c) < tol:
                    continue
                term = f"{c:.4f}"
                for var, power in zip(var_names, exp):
                    if power > 0:
                        term += f"*{var}" + (f"**{power}" if power > 1 else "")
                terms.append(term)
            eqs.append(" + ".join(terms) if terms else "0")
        return eqs


def train_simple_polynomial_ode(
    model,
    y0s,
    t,
    true_ys,
    method="dopri5",
    device="cpu",
    lr=1e-2,
    epochs=2000,
    plateau_window=100,
    plateau_tol=1e-3
):
    y0s = y0s.to(device)
    t = t.to(device)
    true_ys = true_ys.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=50
    )
    loss_fn = nn.MSELoss()

    loss_history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        model.reset_nfe()

        pred_ys = odeint(model, y0s, t, rtol=1e-5, atol=1e-6, method=method)
        pred_ys = pred_ys.permute(1, 0, 2)

        loss = loss_fn(pred_ys, true_ys)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        current_loss = loss.item()
        loss_history.append(current_loss)

        if epoch % 50 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch:4d}, loss: {current_loss:.6e}, nfe: {model.get_nfe()}, lr: {lr_now:.2e}")

        if current_loss < 1e-8:
            print(f"converged at epoch {epoch}")
            break

        if epoch >= plateau_window:
            recent_losses = loss_history[-plateau_window:]
            avg_recent_loss = sum(recent_losses) / len(recent_losses)
            rel_change = abs(current_loss - avg_recent_loss) / (abs(avg_recent_loss) + 1e-12)

            if rel_change < plateau_tol:
                print(
                    f"loss plateau detected at epoch {epoch}: "
                    f"relative change {rel_change:.3e} over last {plateau_window} epochs"
                )
                break

    return model, loss_history


def compute_vector_field_mse(model, problem, device, grid_min=-2.0, grid_max=2.0, n_grid=40):
    if problem_configs[problem]["input_dim"] != 2:
        return None

    x = torch.linspace(grid_min, grid_max, n_grid)
    y = torch.linspace(grid_min, grid_max, n_grid)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    with torch.no_grad():
        learned = model(None, points).cpu()
        true = torch.stack([problem(None, p) for p in points.cpu()]).cpu()

    rel = ((learned - true) ** 2).sum(dim=1) / (true ** 2).sum(dim=1).clamp(min=1e-12)
    return rel.mean().item()


def plot_vector_field_heatmap(model, problem, true_ys, eqs, device, grid_min=-2.0, grid_max=2.0, n_grid=60):
    if problem_configs[problem]["input_dim"] != 2:
        return

    x = torch.linspace(grid_min, grid_max, n_grid)
    y = torch.linspace(grid_min, grid_max, n_grid)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    with torch.no_grad():
        learned = model(None, points).cpu()
        true = torch.stack([problem(None, p) for p in points.cpu()]).cpu()

    error = torch.sqrt(((learned - true) ** 2).sum(dim=1)).view(X.shape)
    rel_mse = (((learned - true) ** 2).sum(dim=1) / (true ** 2).sum(dim=1).clamp(min=1e-12)).mean().item()

    eq_x = eqs[0].replace("**", "^").replace("*", "").replace("+ -", " - ")
    eq_y = eqs[1].replace("**", "^").replace("*", "").replace("+ -", " - ")

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(
        X.numpy(),
        Y.numpy(),
        error.numpy(),
        levels=50,
        cmap="viridis",
        vmin=0,
        vmax=1
    )
    plt.colorbar(contour, label="error magnitude")

    square_x = [-1, 1, 1, -1, -1]
    square_y = [-1, -1, 1, 1, -1]
    plt.plot(square_x, square_y, linewidth=2.5)

    for traj in true_ys:
        traj_np = traj.detach().cpu().numpy()
        plt.plot(traj_np[:, 0], traj_np[:, 1], "w-", linewidth=1.2, alpha=0.7)

    plt.xlabel(
        "$x$\n"
        rf"$\dot{{x}} = {eq_x}$"
        "\n"
        rf"$\dot{{y}} = {eq_y}$",
        fontsize=8,
        labelpad=24
    )
    plt.ylabel("y")
    plt.title(f"vector field error magnitude (relative mse={rel_mse:.4f})")
    plt.tight_layout()
    plt.show()


def main():
    # choose the same problem/config structure as your PNODE code
    problem = data.pitchfork
    cfg = problem_configs[problem]

    t0, t1, N = cfg["t0"], cfg["t1"], cfg["N"]
    input_dim, output_dim = cfg["input_dim"], cfg["output_dim"]
    method = cfg["method"]
    noise_level = cfg["noise_level"]
    lr = cfg["lr"]
    to_normalize = cfg["to_normalize"]
    initial_conditions = cfg["initial_conditions"]

    # choose baseline polynomial degree here
    # degree=2 corresponds to q(x; p) = p0 + p1 x + p2 x^2 in 1d
    # in higher dimensions this becomes all monomials up to degree 2
    baseline_degree = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y0s = torch.stack(initial_conditions).to(device)
    t_train = torch.linspace(t0, t1, N, device=device)

    true_ys = []
    for y0 in y0s:
        _, y = data.gen_trajectory(
            t0=t0,
            t1=t1,
            steps=t_train.shape[0],
            y0=y0,
            device=device,
            problem=problem
        )
        true_ys.append(y)
    true_ys = torch.stack(true_ys, dim=0)

    scale = None
    if to_normalize:
        scale = true_ys.abs().amax(dim=(0, 1)).clamp(min=1e-8)
        true_ys = true_ys / scale
        y0s = y0s / scale

    with torch.no_grad():
        std = noise_level * true_ys.abs().mean()
        noise = std * torch.randn_like(true_ys)
        true_ys_noisy = true_ys + noise

    model = SimplePolynomialODE(
        input_dim=input_dim,
        output_dim=output_dim,
        degree=baseline_degree
    ).to(device)

    print("training simple polynomial baseline...")
    model, loss_history = train_simple_polynomial_ode(
        model,
        y0s,
        t_train,
        true_ys_noisy,
        method=method,
        device=device,
        lr=lr,
        epochs=500
    )

    eqs = model.get_equations(var_names=["x", "y", "z"][:input_dim])

    print("\nlearned baseline equations:")
    if input_dim == 2:
        print("dx/dt =", eqs[0])
        print("dy/dt =", eqs[1])
    elif input_dim == 3:
        print("dx/dt =", eqs[0])
        print("dy/dt =", eqs[1])
        print("dz/dt =", eqs[2])

    pred_y0 = y0s[0]
    pred_path = odeint(model, pred_y0, t_train, rtol=1e-5, atol=1e-6).detach().cpu().numpy()
    true_path = true_ys[0].detach().cpu().numpy()
    noisy_path = true_ys_noisy[0].detach().cpu().numpy()

    final_train_loss = loss_history[-1]
    print(f"\nfinal trajectory mse loss = {final_train_loss:.6e}")

    vf_mse = compute_vector_field_mse(model, problem, device)
    if vf_mse is not None:
        print(f"vector field relative mse on [-2,2]^2 grid = {vf_mse:.6e}")

    plt.figure(figsize=(10, 5))
    if input_dim == 2:
        plt.plot(true_path[:, 0], true_path[:, 1], label="true trajectory", linewidth=2)
        plt.plot(pred_path[:, 0], pred_path[:, 1], "--", label="simple polynomial baseline", linewidth=2)
        plt.scatter(noisy_path[:, 0], noisy_path[:, 1], s=5, alpha=0.5, label="noisy data")
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        plt.plot(true_path[:, 0], true_path[:, 2], label="true trajectory", linewidth=2)
        plt.plot(pred_path[:, 0], pred_path[:, 2], "--", label="simple polynomial baseline", linewidth=2)
        plt.scatter(noisy_path[:, 0], noisy_path[:, 2], s=5, alpha=0.5, label="noisy data")
        plt.xlabel("x")
        plt.ylabel("z")

    plt.title("trajectory fit: simple polynomial baseline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.semilogy(loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training loss: simple polynomial baseline")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if input_dim == 2:
        plot_vector_field_heatmap(
            model=model,
            problem=problem,
            true_ys=true_ys,
            eqs=eqs,
            device=device
        )


if __name__ == "__main__":
    main()
