# train.py
import torch
from model import LoggingWrapper
import torch.optim as optim
import torch.nn as nn
from torchdiffeq import odeint

def train_pnode(model, y0s, t, true_ys, method='dopri5', device='cpu', lr=1e-3, epochs=500):
    y0s = y0s.to(device)
    t = t.to(device)
    true_ys = true_ys.to(device)

    B, T, d = true_ys.shape

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    loss_history = []
    nfe_history = []

    for epoch in range(1, epochs + 1):
        
        optimizer.zero_grad()
        model.reset_nfe()

        # wrap model to log f(y(t))
        wrapper = LoggingWrapper(model)

        pred_ys = odeint(wrapper, y0s, t, rtol=1e-5, atol=1e-6, method=method)  
        pred_ys = pred_ys.permute(1, 0, 2)
        loss = loss_fn(pred_ys, true_ys)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        current_loss = loss.item()
        loss_history.append(current_loss)
        nfe_history.append(model.get_nfe())

        # ---- NEW: analyze logged f -----------------
        # if len(wrapper.history) > 0:
        #     ts = [h[0] for h in wrapper.history]
        #     f_vals = [h[2] for h in wrapper.history]

        #     f_max = max(f_vals)
        #     idx = f_vals.index(f_max)
        #     t_max = ts[idx]

        #     f_avg = sum(f_vals) / len(f_vals)
        #     print(f"f_max: {f_max:.4f}, f_avg: {f_avg:.4f}")
        # # --------------------------------------------

        if epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}, Loss: {current_loss:.6f}, NFE: {model.get_nfe()}, LR: {lr_now:.2e}")

        if current_loss < 1e-5:
            print(f'Converged at epoch {epoch}')
            break
        
        # if epoch > 100:
        #     recent_losses = loss_history[-100:]
        #     avg_loss = sum(recent_losses) / len(recent_losses)
        #     rel_change = abs(current_loss - avg_loss) / (avg_loss + 1e-12)
        #     if rel_change < 0.001:
        #         print(f'Loss plateaued (<0.1% change) over last 100 epochs at epoch {epoch}. Stopping.')
        #         break

    return model, loss_history, nfe_history
