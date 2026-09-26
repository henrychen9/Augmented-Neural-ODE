import torch

@torch.no_grad()
def feature_space_addition_test(model, trials=200):
    model.eval()
    device = next(model.parameters()).device
    poly_dim = model.poly.exponents.shape[0]

    z0 = torch.zeros(poly_dim, device=device)
    f0 = model.net(z0)

    max_rel_err = 0.0
    worst = None

    for _ in range(trials):
        z1 = torch.randn(poly_dim, device=device)
        z2 = torch.randn(poly_dim, device=device)

        lhs = model.net(z1 + z2)
        rhs = model.net(z1) + model.net(z2) - f0

        diff = (lhs - rhs).norm()
        scale = max(lhs.norm().item(), rhs.norm().item(), 1.0)
        rel_err = diff.item() / scale

        if rel_err > max_rel_err:
            max_rel_err = rel_err
            worst = (rel_err,)

    return max_rel_err, worst


@torch.no_grad()
def feature_space_scalar_mult_test(model, trials=200):
    model.eval()
    device = next(model.parameters()).device
    poly_dim = model.poly.exponents.shape[0]

    z0 = torch.zeros(poly_dim, device=device)
    f0 = model.net(z0)

    max_rel_err = 0.0
    worst = None

    for _ in range(trials):
        z = torch.randn(poly_dim, device=device)
        a = torch.randn((), device=device)

        lhs = model.net(a * z)
        rhs = a * model.net(z) + (1 - a) * f0

        diff = (lhs - rhs).norm()
        scale = max(lhs.norm().item(), rhs.norm().item(), 1.0)
        rel_err = diff.item() / scale

        if rel_err > max_rel_err:
            max_rel_err = rel_err
            worst = (a.item(), rel_err)

    return max_rel_err, worst


# usage
from model import PolynomialODE

model = PolynomialODE(input_dim=2, hidden_dims=[64, 64], degree=3)

# override feature library
custom_exps = torch.tensor([
    [1, 0],   # x
    [0, 1],   # v
    [2, 1],   # x^2 v
], dtype=torch.float32)

model.set_exponents(custom_exps)

add_err, add_worst = feature_space_addition_test(model)
mul_err, mul_worst = feature_space_scalar_mult_test(model)

print("Addition test max rel err:", add_err)
print("Addition test worst:", add_worst)

print("Scalar-mult test max rel err:", mul_err)
print("Scalar-mult test worst (a, err):", mul_worst)
