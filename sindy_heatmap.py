import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import data
from configs import problem_configs
from model import PolynomialFeatures
pysindy


# ==================================================
# deterministic seeding
# ==================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==================================================
# exponent generator
# ==================================================
def generate_expo(dim, degree):
    exponents = []
    for powers in itertools.product(range(degree + 1), repeat=dim):
        if sum(powers) <= degree:
            exponents.append(powers)
    return torch.tensor(exponents, dtype=torch.float32)


# ==================================================
# SINDy STLSQ
# ==================================================
def sindy_stlsq(Phi, dXdt, lam=0.05, max_iter=10):

    Xi = np.linalg.lstsq(Phi, dXdt, rcond=None)[0]

    for _ in range(max_iter):

        small = np.abs(Xi) < lam
        Xi[small] = 0

        for i in range(dXdt.shape[1]):
            big = Xi[:, i] != 0

            if np.sum(big) == 0:
                continue

            Xi[big, i] = np.linalg.lstsq(
                Phi[:, big],
                dXdt[:, i],
                rcond=None
            )[0]

    return Xi


# ==================================================
# MAIN
# ==================================================
def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    problem = data.pitchfork
    cfg = problem_configs[problem]

    t0, t1, N = cfg["t0"], cfg["t1"], cfg["N"]
    input_dim = cfg["input_dim"]
    noise_level = cfg["noise_level"]
    degree = cfg["degree"]

    y0s = torch.stack(cfg["initial_conditions"]).to(device)

    dt = (t1 - t0)/(N-1)

    # ==================================================
    # trajectory generation (TRUE)
    # ==================================================
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

    true_ys = torch.stack(true_ys)


    # ==================================================
    # normalization (same as PNODE)
    # ==================================================
    to_normalize = cfg["to_normalize"]

    if to_normalize:
        scale = true_ys.abs().amax(dim=(0, 1)).clamp(min=1e-8)
        true_ys = true_ys / scale
        y0s = y0s / scale


    # ==================================================
    # noisy observations (same as PNODE)
    # ==================================================
    with torch.no_grad():
        std = noise_level * true_ys.abs().mean()
        noise = std * torch.randn_like(true_ys)
        true_ys_noisy = true_ys + noise


    # ==================================================
    # Savitzky-Golay smoothing + derivative estimation
    # ==================================================
    noisy_np = true_ys_noisy.cpu().numpy()

    window_length = 11
    polyorder = 3

    # smooth trajectories
    smooth_np = savgol_filter(
        noisy_np,
        window_length=window_length,
        polyorder=polyorder,
        axis=1
    )

    # estimate derivatives directly from local polynomial fits
    dydt_np = savgol_filter(
        noisy_np,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=dt,
        axis=1
    )

    states = torch.tensor(
        smooth_np,
        dtype=true_ys.dtype,
        device=device
    )

    dydt = torch.tensor(
        dydt_np,
        dtype=true_ys.dtype,
        device=device
    )

    Xtrain = states.reshape(-1, input_dim)
    dXdt = dydt.reshape(-1, input_dim)


    # ==================================================
    # polynomial library
    # ==================================================
    poly = PolynomialFeatures(
        input_dim=input_dim,
        degree=degree
    ).to(device)

    poly.exponents = generate_expo(
        input_dim,
        degree
    ).to(device)

    Phi = poly(Xtrain).cpu().numpy()
    dXdt_np = dXdt.cpu().numpy()


    # ==================================================
    # SINDy fit
    # ==================================================
    Xi = sindy_stlsq(
        Phi,
        dXdt_np,
        lam=0.05,
        max_iter=10
    )

    print("active terms:",
          np.sum(np.abs(Xi)>0))


    # ==================================================
    # grid evaluation
    # ==================================================
    x = torch.linspace(-2,2,40)
    y = torch.linspace(-2,2,40)

    X,Y = torch.meshgrid(x,y,indexing="ij")

    grid = torch.stack(
        [X.flatten(),Y.flatten()],
        dim=1
    ).to(device)

    Phi_grid = poly(grid).cpu().numpy()

    pred = Phi_grid @ Xi

    U_pred = torch.tensor(pred[:,0])
    V_pred = torch.tensor(pred[:,1])


    # ==================================================
    # true field (evaluation only)
    # ==================================================
    with torch.no_grad():
        true_field = torch.stack(
            [problem(None,p.cpu()) for p in grid])

    U_true = true_field[:,0]
    V_true = true_field[:,1]


    # ==================================================
    # error + MSE inside [-1,1]
    # ==================================================
    mask = (
        (grid[:,0]>=-1)&(grid[:,0]<=1)&
        (grid[:,1]>=-1)&(grid[:,1]<=1)
    )

    pred_vec = torch.stack([U_pred,V_pred],dim=1)
    true_vec = torch.stack([U_true,V_true],dim=1)

    mse = (
        ((pred_vec[mask] - true_vec[mask]) ** 2).sum(dim=1)
        /
        (true_vec[mask] ** 2).sum(dim=1).clamp(min=1e-12)
    ).mean().item()

    print(f"MSE (-1,1 box) = {mse:.6f}")


    error = torch.sqrt(
        (U_pred-U_true)**2 +
        (V_pred-V_true)**2
    ).view(X.shape)

    print("\nRecovered equations:")

    exponents = poly.exponents.cpu().numpy()

    for eq in range(input_dim):
        terms = []

        for coeff, powers in zip(Xi[:, eq], exponents):
            if coeff == 0:
                continue

            term = f"{coeff:.4f}"

            for i, power in enumerate(powers):
                if power > 0:
                    term += f"*x{i+1}^{int(power)}"

            terms.append(term)

        print(f"dx{eq+1}/dt = " + " + ".join(terms))


    # ==================================================
    # heatmap
    # ==================================================
    plt.figure(figsize=(8,6))

    plt.contourf(
        X,
        Y,
        error,
        levels=50,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.colorbar(label="Error")

    square_x=[-1,1,1,-1,-1]
    square_y=[-1,-1,1,1,-1]

    plt.plot(square_x,square_y,
             color="orange",linewidth=2.5)


    # ==================================================
    # overlay TRUE trajectories
    # ==================================================
    for traj in true_ys:
        traj_np = traj.cpu().numpy()
        plt.plot(
            traj_np[:,0],
            traj_np[:,1],
            'w-',alpha=0.6,linewidth=1.2
        )

    plt.title(f"SINDy   MSE={mse:.6f}")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()