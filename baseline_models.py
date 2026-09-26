import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
import data
from configs import problem_configs
from model import PolynomialFeatures


# ==============================
# deterministic seeding
# ==============================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==============================
# exponent generator
# ==============================
def generate_expo(dim, degree):
    exponents = []
    for powers in itertools.product(range(degree + 1), repeat=dim):
        if sum(powers) <= degree:
            exponents.append(powers)
    return torch.tensor(exponents, dtype=torch.float32)


# ==============================
# MAIN
# ==============================
def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    problem = data.pitchfork
    cfg = problem_configs[problem]

    t0, t1, N = cfg["t0"], cfg["t1"], cfg["N"]
    input_dim = cfg["input_dim"]
    noise_level = cfg["noise_level"]
    to_normalize = cfg["to_normalize"]

    degree = 3

    initial_conditions = cfg["initial_conditions"]

    y0s = torch.stack(initial_conditions).to(device)
    t_train = torch.linspace(t0, t1, N, device=device)

    dt = (t1 - t0) / (N - 1)

    # ==============================
    # SAME TRAJECTORY GENERATION
    # ==============================
    true_ys = []

    for y0 in y0s:
        _, y = data.gen_trajectory(
            t0=t0,
            t1=t1,
            steps=N,
            y0=y0,
            device=device,
            problem=problem
        )
        true_ys.append(y)

    true_ys = torch.stack(true_ys, dim=0)

    if to_normalize:
        scale = true_ys.abs().amax(dim=(0,1)).clamp(min=1e-8)
        true_ys = true_ys / scale
        y0s = y0s / scale

    # ==============================
    # ADD NOISE
    # ==============================
    with torch.no_grad():
        std = noise_level * true_ys.abs().mean()
        noise = std * torch.randn_like(true_ys)
        noisy = true_ys + noise


    # ==============================
    # NUMERICAL DERIVATIVE
    # ==============================
    # central difference
    dydt = (noisy[:,2:] - noisy[:,:-2]) / (2*dt)
    states = noisy[:,1:-1]

    traj_points = states.reshape(-1, input_dim)
    f_vals = dydt.reshape(-1, input_dim)

    print("training samples:", traj_points.shape[0])


    # ==============================
    # POLYNOMIAL FEATURES
    # ==============================
    poly = PolynomialFeatures(
        input_dim=input_dim,
        degree=degree
    ).to(device)

    poly.exponents = generate_expo(
        input_dim,
        degree
    ).to(device)

    Phi = poly(traj_points)

    Phi_np = Phi.cpu().numpy()
    f_np = f_vals.cpu().numpy()


    # ==============================
    # LEAST SQUARES
    # ==============================
    coeffs_x, *_ = np.linalg.lstsq(
        Phi_np, f_np[:,0], rcond=None)

    coeffs_y, *_ = np.linalg.lstsq(
        Phi_np, f_np[:,1], rcond=None)


    # ==============================
    # GRID EVALUATION
    # ==============================
    x = torch.linspace(-2,2,20)
    y = torch.linspace(-2,2,20)

    X,Y = torch.meshgrid(x,y,indexing="ij")
    grid = torch.stack(
        [X.flatten(),Y.flatten()],
        dim=1).to(device)

    Phi_grid = poly(grid).cpu().numpy()

    U_poly = Phi_grid @ coeffs_x
    V_poly = Phi_grid @ coeffs_y


    # TRUE FIELD (evaluation only)
    with torch.no_grad():
        true_field = torch.stack(
            [problem(None,p.cpu()) for p in grid])

    U_true = true_field[:,0]
    V_true = true_field[:,1]


    # ==============================
    # ERROR HEATMAP
    # ==============================
    error = torch.sqrt(
        (torch.tensor(U_poly)-U_true)**2 +
        (torch.tensor(V_poly)-V_true)**2
    )

    # ==============================
    # relative MSE (same as NODE)
    # ==============================
    pred_field = torch.stack([
        torch.tensor(U_poly),
        torch.tensor(V_poly)
    ], dim=1)

    true_field_vec = torch.stack([
        U_true,
        V_true
    ], dim=1)

    mse_val = (
        ((pred_field - true_field_vec)**2).sum(dim=1)
        /
        (true_field_vec**2).sum(dim=1).clamp(min=1e-12)
    ).mean().item()

    print(f"MSE = {mse_val:.6f}")

    error_grid = error.view(X.shape)

    plt.figure(figsize=(8,6))

    contour = plt.contourf(
        X,Y,error_grid,
        levels=50,
        cmap="viridis")

    plt.colorbar(label="Error")

    square_x=[-1,1,1,-1,-1]
    square_y=[-1,-1,1,1,-1]

    plt.plot(square_x,square_y,
             color="orange",linewidth=2.5)

    plt.title(f"MSE = {mse_val:.6f}")

    plt.xlabel("x")
    plt.ylabel("y")

    # ==============================
    # overlay TRUE (clean) trajectories
    # ==============================
    for traj in true_ys:
        traj_np = traj.cpu().numpy()

        plt.plot(
            traj_np[:,0],
            traj_np[:,1],
            'w-',
            linewidth=1.5,
            alpha=0.7
        )
    plt.tight_layout()
            
    plt.show()


if __name__ == "__main__":
    main()