# configs.py
import torch
import data

problem_configs = {
    data.lorenz: dict(
        t0=0.0, t1=1.0, N=1000,
        input_dim=3, output_dim=3,
        method='dopri5',
        noise_level=0.00,
        lr=1e-2,
        to_normalize=True,
        degree=3,
        initial_conditions=[
            torch.tensor([ 1.0,  1.0,  1.0]),
            torch.tensor([-1.0, -1.0, 20.0]),
            torch.tensor([ 5.0,  5.0,  5.0]),
        ],
    ),

    data.van_der_pol: dict(
        t0=0.0, t1=10.0, N=2000,
        input_dim=2, output_dim=2,
        method='dopri5',
        noise_level=0.10,
        lr=1e-2,
        to_normalize=False,
        degree=3,
        initial_conditions=[
            torch.tensor([-1.0,  0.0]),
            torch.tensor([ 0.0,  1.0]),
            torch.tensor([ 0.0, -1.0]),
            torch.tensor([ 1.0,  0.0]),
        ],
    ),

    data.pitchfork: dict(
        t0=0.0, t1=1.57, N=1000,
        input_dim=2, output_dim=2,
        method='dopri5',
        noise_level=0.10,
        lr=1e-2,
        to_normalize=False,
        degree=3,
        initial_conditions = [
            torch.tensor([x, y])
            for x in torch.linspace(-1.5, 1.5, 4)
            for y in torch.linspace(-1.5, 1.5, 4)
        ]
    ),

    data.pendulum: dict(
        t0=0.0, t1=10.0, N=2000,
        input_dim=2, output_dim=2,
        method='dopri5',
        noise_level=0.00,
        lr=1e-2,
        to_normalize=False,
        degree=3,
        initial_conditions=[
            torch.tensor([-1.5, -1.5]),
            torch.tensor([-1.5,  0.5]),
            torch.tensor([-1.5,  1.5]),
            torch.tensor([ 0.0, -1.5]),
            torch.tensor([ 0.0,  0.0]),
            torch.tensor([ 0.0,  1.5]),
            torch.tensor([ 1.5, -1.5]),
            torch.tensor([ 1.5,  0.0]),
            torch.tensor([ 1.5,  1.5]),
        ],
    ),

    data.sine_oscillator: dict(
        t0=0.0, t1=10.0, N=1000,
        input_dim=2, output_dim=2,
        method='dopri5',
        noise_level=0.10,
        lr=1e-2,
        to_normalize=True,
        degree=3,
        initial_conditions=[
            torch.tensor([-2.0, 0.0]),
            torch.tensor([-1.0, 1.0]),
            torch.tensor([0.0, -1.0]),
            torch.tensor([1.0, 0.5]),
            torch.tensor([2.0, -1.0]),
        ],
    ),

    data.duffing: dict(
        t0=0.0, t1=20.0, N=4000,
        input_dim=2, output_dim=2,
        method='dopri5',
        noise_level=0.00,
        lr=1e-2,
        to_normalize=False,
        degree=3,

        # sample ICs on both sides of the double-well
        initial_conditions=[
            torch.tensor([-1.5,  0.0]),
            torch.tensor([-1.0,  0.5]),
            torch.tensor([-0.5, -0.5]),
            torch.tensor([ 0.0,  1.0]),
            torch.tensor([ 0.5, -1.0]),
            torch.tensor([ 1.0,  0.5]),
            torch.tensor([ 1.5,  0.0]),
        ],
    ),

    data.lotka_volterra: dict(
        t0=0.0, t1=10.0, N=2000,
        input_dim=2, output_dim=2,
        method='dopri5',
        noise_level=0.00,
        lr=1e-2,
        to_normalize=False,
        degree=2,
        initial_conditions=[
            torch.tensor([0.5, 0.5]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([2.0, 1.0]),
            torch.tensor([1.5, 0.3]),
            torch.tensor([0.3, 1.5]),
        ],
    ),

    data.soft_nonlinear: dict(
        t0=0.0, t1=20.0, N=2000,
        input_dim=2, output_dim=2,
        method='dopri5',
        noise_level=0.00,
        lr=1e-3,
        to_normalize=False,
        degree=3,
        initial_conditions=[
            torch.tensor([x, y])
            for x in torch.linspace(-1.5, 1.5, 5)
            for y in torch.linspace(-1.5, 1.5, 5)
        ]
    ),

    data.quartic_duffing: dict(
        t0=0.0, t1=20.0, N=1000,
        input_dim=2, output_dim=2,
        method='dopri5',
        noise_level=0.00,
        lr=1e-2,
        to_normalize=False,
        degree=4,
        initial_conditions=[
            torch.tensor([x, y])
            for x in torch.linspace(-1.5, 1.5, 5)
            for y in torch.linspace(-1.5, 1.5, 5)
        ],
    ),

    data.quintic_duffing: dict(
        t0=0.0,
        t1=20.0,
        N=2000,

        input_dim=2,
        output_dim=2,

        method='dopri5',
        noise_level=0.00,
        lr=1e-2,

        to_normalize=False,
        degree=5,

        initial_conditions=[
            torch.tensor([x, y])
            for x in torch.linspace(-1.5, 1.5, 5)
            for y in torch.linspace(-1.5, 1.5, 5)
        ],
    ),

}
