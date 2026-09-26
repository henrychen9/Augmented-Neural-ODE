# model.py
import torch
import torch.nn as nn
import numpy as np
from sympy import symbols, expand, Matrix, lambdify
import itertools

class PolynomialFeatures(nn.Module):
    """Custom layer to compute polynomial features up to a given degree"""
    def __init__(self, input_dim, degree):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        
        # Generate all possible exponent combinations
        exponents = []
        for d in range(degree + 1):
            exponents.extend(self.generate_exponents(input_dim, d))
        
        self.register_buffer('exponents', torch.tensor(exponents, dtype=torch.float32))
        
    def generate_exponents(self, dim, degree):
        # degree = 2

        """Generate all exponent combinations for given dimension and degree"""
        if dim == 0:
            return [[]]
        if dim == 1:
            # RETURN ALL VALUES <= degree, not just [degree]
            return [[i] for i in range(degree + 1)]

        exponents = []
        for i in range(degree + 1):
            # recurse with SAME degree, not degree - i
            for e in self.generate_exponents(dim - 1, degree - i):
                exponents.append([i] + e)
        return exponents

        return [[0, 1], [1, 0], [0, 1], [3, 0], [4, 0]]
    
    def set_exponents(self, new_exps):
        # keep as plain tensor; we always move to x.device in forward
        self.exponents = new_exps

        
    def forward(self, x):
        # Ensure x has the right shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.size(0)
        
        # Compute polynomial features: x₁^e₁ * x₂^e₂ * ... 
        # Use proper broadcasting
        exponents_expanded = self.exponents.unsqueeze(0).to(x.device)  # [1, num_terms, input_dim]
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Compute each term: x^exponent
        terms = x_expanded ** exponents_expanded
        
        # Multiply across the input dimension to get the polynomial features
        features = torch.prod(terms, dim=2)  # [batch_size, num_terms]
        
        return features

class PolynomialODE(nn.Module):
    """Polynomial Neural ODE"""
    def __init__(self, input_dim, hidden_dims, degree, output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = output_dim if output_dim else input_dim
        
        # Polynomial feature expansion
        self.poly = PolynomialFeatures(input_dim, degree)
        poly_dim = len(self.poly.exponents)
        
        # Neural network to learn coefficients for polynomial terms
        layers = []
        prev_dim = poly_dim
        # prev_dim = 2
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.GELU())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.net = nn.Sequential(*layers)
        
        # For tracking function evaluations
        self.nfe = 0

    def set_exponents(self, new_exps):
        """
        Update the polynomial feature library AND resize the first Linear layer
        so that its input dimension matches the new number of terms.
        """
        # update feature exponents
        self.poly.set_exponents(new_exps)
        poly_dim = new_exps.shape[0]

        # grab existing layers
        layers = list(self.net.children())
        first_linear = layers[0]
        assert isinstance(first_linear, nn.Linear)

        # device of current parameters
        device = first_linear.weight.device

        # new first layer with correct in_features
        new_first = nn.Linear(poly_dim, first_linear.out_features).to(device)

        # optional: copy overlapping weights so we don't completely reset
        with torch.no_grad():
            old_in = first_linear.in_features
            n_copy = min(old_in, poly_dim)
            if n_copy > 0:
                new_first.weight[:, :n_copy] = first_linear.weight[:, :n_copy]
            if first_linear.bias is not None:
                new_first.bias.copy_(first_linear.bias)

        # replace layer 0 and rebuild Sequential
        layers[0] = new_first
        self.net = nn.Sequential(*layers)
    
    def feature_transform(self, x):
        # Default: polynomial expansion
        return self.poly(x)
        
        # # extract variables
        # X = x[:, 0:1]   # shape [batch_size, 1]
        # Y = x[:, 1:2]   # shape [batch_size, 1]

        # # build your feature library
        # return torch.cat([torch.sin(X), Y], dim=1)
        
    def forward(self, t, x):
        self.nfe += 1
        
        # Handle both single and batch inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
            was_1d = True
        else:
            was_1d = False

        # Learn coefficients for features
        features = self.feature_transform(x)
        result = self.net(features)
        
        # Return to original shape if needed
        if was_1d:
            result = result.squeeze(0)
            
        return result
    
    def reset_nfe(self):
        self.nfe = 0
        
    def get_nfe(self):
        return self.nfe
    
    def get_symbolic_equation(self, var_names=None):
        """Extract symbolic equation from the trained model"""
        if var_names is None:
            var_names = [f'x{i}' for i in range(self.input_dim)]
        
        # Create symbols for variables
        vars = symbols(var_names)
        
        # Get the weights from the final layer
        final_layer = self.net[-1]
        weights = final_layer.weight.detach().cpu().numpy()
        bias = final_layer.bias.detach().cpu().numpy() if final_layer.bias is not None else np.zeros(self.output_dim)
        
        # Create symbolic expressions
        equations = []
        for i in range(self.output_dim):
            eq = 0
            for j, exp in enumerate(self.poly.exponents):
                term = weights[i, j]
                for k, e in enumerate(exp):
                    if e > 0:
                        term *= vars[k] ** int(e)
                eq += term
            eq += bias[i]
            equations.append(expand(eq))
        
        return equations

class LoggingWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.history = []  # (t, ||y||, ||f||)

    def forward(self, t, y):
        f = self.base(t, y)

        with torch.no_grad():
            norm_y = y.norm(dim=1).mean().item()
            norm_f = f.norm(dim=1).mean().item()
            self.history.append((float(t), norm_y, norm_f))

        return f

