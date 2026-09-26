# main.py
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from torchdiffeq import odeint
import data
from model import PolynomialODE
from model import PolynomialFeatures
from train import train_pnode
from configs import problem_configs

def generate_expo(dim, degree):
    exponents = []
    for powers in itertools.product(range(degree + 1), repeat=dim):
        if sum(powers) <= degree:
            exponents.append(powers)
    return torch.tensor(exponents, dtype=torch.float32)


def main():
    # Parameters
    problem = data.pitchfork
    cfg = problem_configs[problem]

    t0, t1, N = cfg["t0"], cfg["t1"], cfg["N"]
    input_dim, output_dim = cfg["input_dim"], cfg["output_dim"]
    method = cfg["method"]
    noise_level = cfg["noise_level"]

    lr = cfg["lr"]
    to_normalize = cfg["to_normalize"]
    degree = cfg["degree"]
    initial_conditions = cfg["initial_conditions"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y0s = torch.stack(initial_conditions).to(device)

    # Time grid
    t_train = torch.linspace(t0, t1, N, device=device)

    # Generate true trajectories
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

    if to_normalize:
        scale = true_ys.abs().amax(dim=(0, 1)).clamp(min=1e-8)
        true_ys = true_ys / scale
        y0s = y0s / scale
    
    # === Add 1% Gaussian noise to the true trajectories ===
    with torch.no_grad():
        std = noise_level * true_ys.abs().mean()
        noise = std * torch.randn_like(true_ys)
        true_ys_noisy = true_ys + noise


    while True:
        try:
            model = PolynomialODE(
                input_dim=input_dim,
                hidden_dims=[40, 40],
                degree=degree,
                output_dim=output_dim
            ).to(device)
            
            print("Training Polynomial Neural ODE...")
            model, loss_history, nfe_history = train_pnode(
                model,
                y0s,
                t_train,
                true_ys_noisy,
                method=method,
                device=device,
                lr=lr,
                epochs=1000
            )
            break

        except (AssertionError, RuntimeError) as e:
            print("Training failed:", str(e))
            print("Restarting from epoch 1...\n")

    # === Polynomial fit on one trajectory ===
    print("\nFitting polynomial (deg=3) to one trajectory...")
    N_points = 100000
    use_traj = False

    if use_traj:
        y0 = torch.tensor([0.5, -0.5], device=device)
        t_traj = torch.linspace(t0, t1, N_points, device=device)
        y_traj = odeint(model, y0, t_traj, rtol=1e-5, atol=1e-6)
        traj_points = y_traj
    else:
        traj_points = (torch.rand(N_points, input_dim, device=device) * 2) - 1

    # compute derivatives at each point along trajectory
    with torch.no_grad():
        f_vals = model(None, traj_points)

    # build polynomial features up to degree 3
    poly = PolynomialFeatures(input_dim=input_dim, degree=degree).to(device)
    poly.exponents = generate_expo(input_dim, degree).to(device)
    Phi = poly(traj_points)  # [500, 10]

    # solve least squares for both dx and dv
    Phi_np = Phi.detach().cpu().numpy()
    f_np = f_vals.detach().cpu().numpy()

    coeffs_x, _, _, _ = np.linalg.lstsq(Phi_np, f_np[:, 0], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(Phi_np, f_np[:, 1], rcond=None)
    if input_dim == 3:
        coeffs_z, _, _, _ = np.linalg.lstsq(Phi_np, f_np[:, 2], rcond=None)

    # print polynomial equations
    exps = poly.exponents.cpu().numpy().astype(int)
    var_names = ["x", "y", "z"]

    def coeffs_to_equation(coeffs):
        terms = []
        for c, exp in zip(coeffs, exps):
            if abs(c) < 1e-6:
                continue
            term = f"{c:.4f}"
            for var, power in zip(var_names, exp):
                if power > 0:
                    term += f"*{var}" + (f"**{power}" if power > 1 else "")
            terms.append(term)
        return " + ".join(terms) if terms else "0"

    eq_dx = coeffs_to_equation(coeffs_x)
    eq_dy = coeffs_to_equation(coeffs_y)

    print("dx/dt ≈", eq_dx)
    print("dy/dt ≈", eq_dy)

    if input_dim == 3:
        eq_dz = coeffs_to_equation(coeffs_z)
        print("dz/dt ≈", eq_dz)

    # Test on the first trajectory
    pred_y0 = y0s[0]
    pred_path = odeint(
        model,
        pred_y0,
        t_train,
        rtol=1e-5,
        atol=1e-6
    ).detach().cpu().numpy()

    true_path = true_ys[0].cpu().numpy()
    noisy_path = true_ys_noisy[0].cpu().numpy()

    # ----
    # The entire plotting section remains UNCHANGED.
    # I only fixed indentation so it's legal Python.
    # ----

    if problem == data.lorenz:
        # Plot true vs predicted trajectory
        plt.figure(figsize=(10, 5))
        plt.plot(true_path[:, 0], true_path[:, 2], 'b-', label='True trajectory', linewidth=2)
        plt.plot(pred_path[:, 0], pred_path[:, 2], 'r--', label='PNODE prediction', linewidth=2)
        plt.scatter(noisy_path[:, 0], noisy_path[:, 2], s=5, c='gray', alpha=0.5, label='Noisy data')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.legend()
        plt.title('True ys Noisy ys Predicted Trajectory')
        plt.grid(True)
        plt.show()
        
        # Create a grid for vector field visualization
        x = torch.linspace(-2, 2, 20)
        z = torch.linspace(-2, 2, 20)
        X, Z = torch.meshgrid(x, z, indexing='ij')
        Y_plane = torch.zeros_like(X)  # fix y=0 slice for visualization
        points = torch.stack([X.flatten(), Y_plane.flatten(), Z.flatten()], dim=1).to(device)
        
        # Compute learned vector field
        with torch.no_grad():
            dydt_learned = model(None, points).cpu()
        U_learned, W_learned = dydt_learned[:, 0], dydt_learned[:, 2]
        
        # Compute true vector field
        with torch.no_grad():
            dydt_true = torch.stack([problem(None, p) for p in points.cpu()]).cpu()
        U_true, W_true = dydt_true[:, 0], dydt_true[:, 2]
        
        # Plot vector fields side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True vector field
        ax1.quiver(X, Z, U_true.view(X.shape), W_true.view(Z.shape))
        ax1.set_title('True Vector Field')
        ax1.set_xlabel('x')
        ax1.set_ylabel('z')
        # ax1.set_xlim(-2, 2)
        # ax1.set_ylim(-2, 2)
        
        # Learned vector field
        ax2.quiver(X, Z, U_learned.view(X.shape), W_learned.view(Z.shape))
        ax2.set_title('Learned Vector Field')
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        # ax2.set_xlim(-2, 2)
        # ax2.set_ylim(-2, 2)
        
        plt.tight_layout()
        plt.show()
        
        # Plot error magnitude
        error = torch.sqrt((U_learned - U_true.view(-1))**2 + (W_learned - W_true.view(-1))**2)
        error_grid = error.view(X.shape)

        # compute grid MSE inline
        mse_val = (((dydt_learned - dydt_true)**2).sum(dim=1) / (dydt_true**2).sum(dim=1).clamp(min=1e-12)).mean().item()
        
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Z, error_grid, levels=50, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(contour, label='Error Magnitude')
        square_x = [-1, 1, 1, -1, -1]
        square_y = [-1, -1, 1, 1, -1]
        plt.plot(square_x, square_y, color='orange', linewidth=2.5)


        plt.xlabel(
            "$x$\n"
            rf"$\dot{{x}} = {eq_dx.replace("**", "^").replace("*", "").replace("+ -", " - ")}$"
            "\n"
            rf"$\dot{{y}} = {eq_dy.replace("**", "^").replace("*", "").replace("+ -", " - ")}$"
            "\n"
            rf"$\dot{{z}} = {eq_dz.replace("**", "^").replace("*", "").replace("+ -", " - ")}$",
            fontsize=8,
            labelpad=24
        )

        plt.ylabel('z')
        plt.title(f'Vector Field Error Magnitude (MSE={mse_val:.4f})')
        
        # Overlay training trajectories
        for traj in true_ys:
            traj_np = traj.cpu().numpy()
            plt.plot(traj_np[:, 0], traj_np[:, 2], 'w-', linewidth=1.5, alpha=0.7)

        if (use_traj):
            # overlay the selected PNODE trajectory used for polynomial fit in red
            pn_traj_np = y_traj.detach().cpu().numpy()
            plt.plot(pn_traj_np[:, 0], pn_traj_np[:, 2], 'r-', linewidth=2.0, alpha=0.95, label='computed traj')
            plt.legend(loc='upper right')

            # overlay the TRUE trajectory (same IC/time/steps) in green
            true_traj = odeint(problem, y0, t_traj, rtol=1e-5, atol=1e-6)
            true_np = true_traj.detach().cpu().numpy()
            plt.plot(true_np[:, 0], true_np[:, 2], 'g-', linewidth=2.2, alpha=0.95, zorder=3, label='true traj')
            plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(10, 5))
        plt.plot(true_path[:, 0], true_path[:, 1], 'b-', label='True trajectory', linewidth=2)
        plt.plot(pred_path[:, 0], pred_path[:, 1], 'r--', label='PNODE prediction', linewidth=2)
        plt.scatter(noisy_path[:, 0], noisy_path[:, 1], s=5, c='gray', alpha=0.5, label='Noisy data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('True ys Noisy ys Predicted Trajectory')
        plt.grid(True)
        plt.show()
        
        # # Plot training loss
        # plt.figure(figsize=(10, 5))
        # plt.semilogy(loss_history)
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss (log scale)')
        # plt.title('Training Loss')
        # plt.grid(True)
        # plt.show()
        
        # # Plot NFE over training
        # plt.figure(figsize=(10, 5))
        # plt.plot(nfe_history)
        # plt.xlabel('Epoch')
        # plt.ylabel('NFE')
        # plt.title('Number of Function Evaluations During Training')
        # plt.grid(True)
        # plt.show()
        
        # Extract and print symbolic equation
        # print("\nLearned Symbolic Equations:")
        # equations = model.get_symbolic_equation(var_names=['x', 'y'])
        # for i, eq in enumerate(equations):
        #     print(f"dx{i+1}/dt = {eq}")
        
        # Create a grid for vector field visualization
        x = torch.linspace(-2, 2, 20)
        y = torch.linspace(-2, 2, 20)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
        
        # Compute learned vector field
        with torch.no_grad():
            dydt_learned = model(None, points).cpu()
        U_learned, W_learned = dydt_learned[:, 0], dydt_learned[:, 1]
        
        # Compute true vector field
        with torch.no_grad():
            dydt_true = torch.stack([problem(None, p) for p in points.cpu()]).cpu()
        U_true, W_true = dydt_true[:, 0], dydt_true[:, 1]
        
        # Plot vector fields side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True vector field
        ax1.quiver(X, Y, U_true.view(X.shape), W_true.view(Y.shape))
        ax1.set_title('True Vector Field')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        # ax1.set_xlim(-2, 2)
        # ax1.set_ylim(-2, 2)
        
        # Learned vector field
        ax2.quiver(X, Y, U_learned.view(X.shape), W_learned.view(Y.shape))
        ax2.set_title('Learned Vector Field')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        # ax2.set_xlim(-2, 2)
        # ax2.set_ylim(-2, 2)
        
        plt.tight_layout()
        plt.show()
        
        # Plot error magnitude
        error = torch.sqrt((U_learned - U_true.view(-1))**2 + (W_learned - W_true.view(-1))**2)
        error_grid = error.view(X.shape)

        # compute grid MSE inline
        mse_val = (((dydt_learned - dydt_true)**2).sum(dim=1) / (dydt_true**2).sum(dim=1).clamp(min=1e-12)).mean().item()
        
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, error_grid, levels=50, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(contour, label='Error Magnitude')
        square_x = [-1, 1, 1, -1, -1]
        square_y = [-1, -1, 1, 1, -1]
        plt.plot(square_x, square_y, color='orange', linewidth=2.5)


        plt.xlabel(
            "$x$\n"
            rf"$\dot{{x}} = {eq_dx.replace("**", "^").replace("*", "").replace("+ -", " - ")}$"
            "\n"
            rf"$\dot{{y}} = {eq_dy.replace("**", "^").replace("*", "").replace("+ -", " - ")}$",
            fontsize=8,
            labelpad=24
        )

        plt.ylabel('y')
        plt.title(f'Vector Field Error Magnitude (MSE={mse_val:.4f})')
        
        # Overlay training trajectories
        for traj in true_ys:
            traj_np = traj.cpu().numpy()
            plt.plot(traj_np[:, 0], traj_np[:, 1], 'w-', linewidth=1.5, alpha=0.7)

        if (use_traj):
            # overlay the selected PNODE trajectory used for polynomial fit in red
            pn_traj_np = y_traj.detach().cpu().numpy()
            plt.plot(pn_traj_np[:, 0], pn_traj_np[:, 1], 'r-', linewidth=2.0, alpha=0.95, label='computed traj')
            plt.legend(loc='upper right')

            # overlay the TRUE trajectory (same IC/time/steps) in green
            true_traj = odeint(problem, y0, t_traj, rtol=1e-5, atol=1e-6)
            true_np = true_traj.detach().cpu().numpy()
            plt.plot(true_np[:, 0], true_np[:, 1], 'g-', linewidth=2.2, alpha=0.95, zorder=3, label='true traj')
            plt.legend(loc='upper right')

        # === Second heat map from -1 to 1 ===
        plt.figure(figsize=(8, 6))

        # Create a grid from -1 to 1 for vector field visualization
        x_small = torch.linspace(-1, 1, 20)
        y_small = torch.linspace(-1, 1, 20)
        X_small, Y_small = torch.meshgrid(x_small, y_small, indexing='ij')
        points_small = torch.stack([X_small.flatten(), Y_small.flatten()], dim=1).to(device)

        # Compute learned vector field on small grid
        with torch.no_grad():
            dydt_learned_small = model(None, points_small).cpu()
        U_learned_small, W_learned_small = dydt_learned_small[:, 0], dydt_learned_small[:, 1]

        # Compute true vector field on small grid
        with torch.no_grad():
            dydt_true_small = torch.stack([problem(None, p) for p in points_small.cpu()]).cpu()
        U_true_small, W_true_small = dydt_true_small[:, 0], dydt_true_small[:, 1]

        # Plot error magnitude for small grid
        error_small = torch.sqrt((U_learned_small - U_true_small.view(-1))**2 + (W_learned_small - W_true_small.view(-1))**2)
        error_grid_small = error_small.view(X_small.shape)

        # compute grid MSE for small region
        mse_val_small = (((dydt_learned_small - dydt_true_small)**2).sum(dim=1) / (dydt_true_small**2).sum(dim=1).clamp(min=1e-12)).mean().item()

        contour_small = plt.contourf(X_small, Y_small, error_grid_small, levels=50, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(contour_small, label='Error Magnitude')

        # Plot the square boundary
        square_x = [-1, 1, 1, -1, -1]
        square_y = [-1, -1, 1, 1, -1]
        plt.plot(square_x, square_y, color='orange', linewidth=2.5)

        plt.xlabel(
            "$x$\n"
            rf"$\dot{{x}} = {eq_dx.replace("**", "^").replace("*", "").replace("+ -", " - ")}$"
            "\n"
            rf"$\dot{{y}} = {eq_dy.replace("**", "^").replace("*", "").replace("+ -", " - ")}$",
            fontsize=8,
            labelpad=24
        )

        plt.ylabel('y')
        plt.title(f'Vector Field Error Magnitude [-1,1] (MSE={mse_val_small:.4f})')

        # Overlay training trajectories (only the parts within [-1,1])
        for traj in true_ys:
            traj_np = traj.cpu().numpy()
            # Filter points within [-1,1] range for cleaner display
            mask = (traj_np[:, 0] >= -1) & (traj_np[:, 0] <= 1) & (traj_np[:, 1] >= -1) & (traj_np[:, 1] <= 1)
            if mask.any():
                plt.plot(traj_np[mask, 0], traj_np[mask, 1], 'w-', linewidth=1.5, alpha=0.7)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()