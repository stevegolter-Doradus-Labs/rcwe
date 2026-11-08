import math
import time
import torch
from typing import Tuple, List, Dict, Optional
from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.sandbox.baseline_tf import BaselineCausalTransformer
from rcwe.kernels.accelerator import select_best_device, sync_device, get_device_memory_kb


class HardwareProfilingHarness:
    @staticmethod
    def theoretical_flops(
        seq_len: int, d_model: int, num_bands: int = 16, num_worldlines: int = 3
    ) -> Tuple[float, float]:
        tf_flops = 2.0 * (4.0 * seq_len * (d_model ** 2) + 2.0 * (seq_len ** 2) * d_model)
        rcwe_flops = (
            10.0 * seq_len * d_model * math.log2(max(seq_len, 2))
            + 8.0 * seq_len * (d_model ** 2)
            + 14.0 * seq_len * num_bands * d_model
            + 12.0 * num_worldlines * seq_len * (d_model ** 2)
        )
        return tf_flops / 1e6, rcwe_flops / 1e6

    @staticmethod
    def state_memory_kb(
        seq_len: int, batch_size: int, d_model: int, num_bands: int = 16
    ) -> Tuple[float, float]:
        tf_kv_bytes = 2 * 2 * batch_size * seq_len * d_model * 4
        tf_attn_bytes = batch_size * 4 * (seq_len ** 2) * 4
        tf_total_kb = (tf_kv_bytes + tf_attn_bytes) / 1024.0

        rcwe_psi_bytes = batch_size * num_bands * d_model * 8
        rcwe_fast_w_bytes = (d_model ** 2) * 4
        rcwe_total_kb = (rcwe_psi_bytes + rcwe_fast_w_bytes) / 1024.0
        return tf_total_kb, rcwe_total_kb

    @classmethod
    def run_scaling_sweep(
        cls,
        d_model: int = 64,
        batch_size: int = 2,
        seq_lengths: Optional[List[int]] = None,
        prefer_device: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        if seq_lengths is None:
            seq_lengths = [256, 1024, 2048, 4096]
        device = select_best_device(prefer_device)
        rcwe = ResonantCausalWorldEngine(d_input=32, d_model=d_model, d_action=8, num_worldlines=3).to(device)
        tf_model = BaselineCausalTransformer(d_input=32, d_model=d_model, d_action=8, nhead=4, num_layers=2).to(device)
        rcwe.eval()
        tf_model.eval()

        results = []
        for L in seq_lengths:
            tf_mflops, rcwe_mflops = cls.theoretical_flops(L, d_model)
            tf_state_kb, rcwe_state_kb = cls.state_memory_kb(L, batch_size, d_model)

            x = torch.randn(batch_size, L, 32, device=device)
            a = torch.randn(batch_size, L, 8, device=device)

            with torch.no_grad():
                _ = tf_model(x, a)
                _ = rcwe(x, a, plastic_runtime=False)
            sync_device(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(3):
                    out_tf = tf_model(x, a)
            sync_device(device)
            tf_ms = ((time.perf_counter() - t0) / 3.0) * 1000.0

            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(L, device=device)
            mask_kb = (causal_mask.numel() * causal_mask.element_size()) / 1024.0
            kv_act_kb = (2 * 2 * batch_size * L * d_model * out_tf.element_size()) / 1024.0
            tf_measured_tensor_kb = mask_kb + kv_act_kb

            t1 = time.perf_counter()
            with torch.no_grad():
                for _ in range(3):
                    out_rcwe = rcwe(x, a, plastic_runtime=False)
            sync_device(device)
            rcwe_ms = ((time.perf_counter() - t1) / 3.0) * 1000.0

            psi_tensor = out_rcwe["standing_field"]
            measured_psi_kb = (psi_tensor.numel() * psi_tensor.element_size()) / 1024.0
            hw_alloc_kb = get_device_memory_kb(device)

            results.append({
                "seq_len": float(L),
                "tf_mflops": tf_mflops,
                "rcwe_mflops": rcwe_mflops,
                "tf_kb": tf_state_kb,
                "rcwe_kb": rcwe_state_kb,
                "measured_psi_kb": measured_psi_kb,
                "hw_device_alloc_kb": hw_alloc_kb,
                "tf_latency_ms": tf_ms,
                "rcwe_latency_ms": rcwe_ms,
                "tf_peak_alloc_kb": tf_measured_tensor_kb,
                "rcwe_peak_alloc_kb": measured_psi_kb,
                "compression_ratio": tf_state_kb / rcwe_state_kb,
            })
        return results
