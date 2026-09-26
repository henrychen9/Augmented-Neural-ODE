# data.py
import torch
from torchdiffeq import odeint

def linear(t, y):
    return torch.stack([y[1], -y[0]])

def damped(t, y, omega=1.0, zeta=0.1):
    return torch.stack([y[1], -omega**2 * y[0] - 2*zeta*omega * y[1]])

def van_der_pol(t, y):
    x, v = y[0], y[1]
    dx = v
    dv = (1 - x**2) * v - x
    return torch.stack([dx, dv])

def rayleigh(t, y, epsilon=1.0):
    x, v = y[0], y[1]
    dx = v
    dv = -x - epsilon * (v**3 - v)
    return torch.stack([dx, dv])

def gen_trajectory(t0, t1, steps=1000, y0=None, device='cpu', problem=linear):
    t = torch.linspace(t0, t1, steps, device=device)
    if y0 is None:
        y0 = torch.tensor([2., 0.], device=device)
    y = odeint(problem, y0, t)
    return t, y

def pitchfork(t, y):
    """
    Bistable pitchfork (double-well) system:
        dx/dt = x - x^3
        dy/dt = -y
    Fixed points: (-1, 0), (0, 0), (1, 0)
      - (-1, 0) and (1, 0) are stable attractors
      - (0, 0) is an unstable saddle along x
    """
    x, v = y[0], y[1]
    dx = x - x**3
    dv = -v
    return torch.stack([dx, dv])

def lorenz(t, y, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Lorenz chaotic system:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z
    Chaotic for classic params (sigma=10, rho=28, beta=8/3)
    """
    x, y_, z = y
    dx = sigma * (y_ - x)
    dy = x * (rho - z) - y_
    dz = x * y_ - beta * z
    return torch.stack([dx, dy, dz])

def pendulum(t, y):
    x, v = y
    dx = v
    dv = -torch.sin(x)
    return torch.stack([dx, dv])

def sine_oscillator(t, y):
    x, v = y
    dx = v
    dv = -torch.sin(x) - 0.3*v
    return torch.stack([dx, dv])

def duffing(t, y):
    """
    Duffing oscillator (conservative double-well):
        dx/dt = y
        dy/dt = -alpha * x - beta * x^3

    With alpha = -1, beta = 1:
        dx/dt = y
        dy/dt = x - x^3   (double well)
    """
    x, v = y[0], y[1]
    dx = v
    dv = x - x**3
    return torch.stack([dx, dv])

def lotka_volterra(t, y):
    x, v = y[0], y[1]
    dx = x - x * v
    dv = -v + x * v
    return torch.stack([dx, dv])

def soft_nonlinear(t, y):
    x, v = y
    dx = v
    dv = -x - x**3
    return torch.stack([dx, dv])

def quartic_duffing(t, y, delta=0.2, alpha=-1.0, beta=1.0, gamma=0.2):
    """
    quartic duffing oscillator (degree-4 nonlinearity):
        dx/dt = v
        dv/dt = -0.2*v + x - x^3 - 0.2*x^4

    """
    x, v = y[0], y[1]
    dx = v
    dv = -delta * v - alpha * x - beta * x**3 - gamma * x**4
    return torch.stack([dx, dv])

def quintic_duffing(t, y):
    """
    quintic duffing oscillator:
        dx/dt = v
        dv/dt = x - x^3 - 0.2*x^5

    symmetric double-well system
    """

    x, v = y[0], y[1]

    dx = v
    dv = x - x**3 - 0.2 * x**5

    return torch.stack([dx, dv])