import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.sandbox.baseline_tf import BaselineCausalTransformer
from rcwe.kernels.accelerator import select_best_device, sync_device


class SymbolicConservationTask:
    """
    Symbolic Algebraic Conservation & Invariant Reasoning Task.
    Tracks a 16-register conserved symbolic ledger q_t under directed algebraic
    operators do(a_t). Enforces exact sum invariance: sum(q_t) == C_0 under valid
    symbolic steps, and injects mid-stream rule mutations to test online plasticity.
    """
    def __init__(self, num_registers: int = 16, device: torch.device = torch.device("cpu")):
        self.N = num_registers
        self.device = device

        G = torch.zeros(num_registers, num_registers, device=device)
        for i in range(num_registers):
            G[i, (i + 1) % num_registers] = 0.35
            G[(i + 1) % num_registers, i] = -0.35
            G[i, (i + 3) % num_registers] = -0.15
            G[(i + 3) % num_registers, i] = 0.15
        self.G_conserved = G

        R = torch.zeros(num_registers, 8, device=device)
        for k in range(8):
            R[2 * k, k] = 0.50
            R[2 * k + 1, k] = -0.50
        self.R_opcodes = R

    def generate_batch(
        self, batch_size: int = 8, seq_len: int = 16, rule_regime: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(batch_size, self.N, device=self.device) * 0.5
        q = q - q.mean(dim=-1, keepdim=True) + (10.0 / self.N)
        p = torch.zeros(batch_size, self.N, device=self.device)

        xs, acts, ys = [], [], []
        for t in range(seq_len):
            op = torch.randn(batch_size, 8, device=self.device) * 0.4
            x_t = torch.cat([q, p], dim=-1)

            delta_q = torch.matmul(q, self.G_conserved.t()) + torch.matmul(op, self.R_opcodes.t())

            if rule_regime == 1:
                delta_q = torch.roll(delta_q, shifts=2, dims=-1)
                delta_q[:, :8] += 0.25
                delta_q[:, 8:] -= 0.25
            elif rule_regime == 2:
                delta_q[:, 4:8] -= 0.45

            q_next = q + 0.1 * delta_q
            p_next = delta_q
            y_t = torch.cat([q_next, p_next], dim=-1)

            xs.append(x_t)
            acts.append(op)
            ys.append(y_t)
            q, p = q_next, p_next

        return torch.stack(xs, dim=1), torch.stack(acts, dim=1), torch.stack(ys, dim=1)


def run_symbolic_reasoning_benchmark(seed: int = 42, epochs: int = 45) -> Dict[str, float]:
    torch.manual_seed(seed)
    device = select_best_device()
    task = SymbolicConservationTask(num_registers=16, device=device)

    rcwe = ResonantCausalWorldEngine(d_input=32, d_model=64, d_action=8, num_worldlines=3).to(device)
    tf_model = BaselineCausalTransformer(d_input=32, d_model=64, d_action=8, nhead=4, num_layers=2).to(device)

    opt_rcwe = torch.optim.AdamW(rcwe.parameters(), lr=4e-3)
    opt_tf = torch.optim.AdamW(tf_model.parameters(), lr=4e-3)

    rcwe.train()
    tf_model.train()

    for _ in range(epochs):
        opt_rcwe.zero_grad()
        opt_tf.zero_grad()
        loss_r_total = torch.tensor(0.0, device=device)
        loss_t_total = torch.tensor(0.0, device=device)

        for reg in range(3):
            x_s, a_s, y_s = task.generate_batch(batch_size=6, seq_len=12, rule_regime=reg)
            out_r = rcwe(x_s, a_s, plastic_runtime=False, y_target=y_s)

            true_delta = (y_s - x_s) / 0.1
            pred_delta_r = (out_r["output"] - x_s) / 0.1
            branch_delta = (out_r["branch_predictions"][reg] - x_s) / 0.1

            sum_err_r = (out_r["output"][..., :16].sum(dim=-1) - y_s[..., :16].sum(dim=-1)).abs().mean()
            loss_r_total = (
                loss_r_total
                + F.mse_loss(pred_delta_r, true_delta)
                + F.mse_loss(branch_delta, true_delta)
                + 0.25 * sum_err_r
            )

            out_t = tf_model(x_s, a_s)
            loss_t_total = loss_t_total + F.mse_loss((out_t - x_s) / 0.1, true_delta)

        loss_r_total.backward()
        torch.nn.utils.clip_grad_norm_(rcwe.parameters(), 2.0)
        opt_rcwe.step()

        loss_t_total.backward()
        opt_tf.step()

    sync_device(device)
    rcwe.eval()
    tf_model.eval()

    with torch.no_grad():
        x_val, a_val, y_val = task.generate_batch(batch_size=8, seq_len=16, rule_regime=0)
        out_r_val = rcwe(x_val, a_val, plastic_runtime=False, y_target=y_val)["output"]
        out_t_val = tf_model(x_val, a_val)

        rcwe_sum_violation = float((out_r_val[..., :16].sum(dim=-1) - y_val[..., :16].sum(dim=-1)).abs().mean().item())
        tf_sum_violation = float((out_t_val[..., :16].sum(dim=-1) - y_val[..., :16].sum(dim=-1)).abs().mean().item())

        rcwe.reset_runtime_adaptation()
        x_mut, a_mut, y_mut = task.generate_batch(batch_size=4, seq_len=20, rule_regime=1)
        rule_shift = torch.zeros(4, 1, 32, device=device)
        rule_shift[:, :, :8] = 0.35
        rule_shift[:, :, 8:16] = -0.35

        standing_field = None
        r_mut_mses, t_mut_mses = [], []
        for t in range(20):
            xt = x_mut[:, t : t + 1, :]
            at = a_mut[:, t : t + 1, :]
            yt = y_mut[:, t : t + 1, :] + rule_shift

            res_r = rcwe(xt, at, standing_field=standing_field, plastic_runtime=True, y_target=yt)
            standing_field = res_r["standing_field"]
            res_t = tf_model(xt, at)

            r_mut_mses.append(float(F.mse_loss(res_r["output"], yt).item()))
            t_mut_mses.append(float(F.mse_loss(res_t, yt).item()))

    return {
        "rcwe_symbolic_conservation_error": rcwe_sum_violation,
        "tf_symbolic_conservation_error": tf_sum_violation,
        "rcwe_rule_mutation_mse": sum(r_mut_mses[-5:]) / 5.0,
        "tf_rule_mutation_mse": sum(t_mut_mses[-5:]) / 5.0,
    }
