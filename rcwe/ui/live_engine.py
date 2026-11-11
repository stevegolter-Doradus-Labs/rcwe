"""
The engine room behind Mission Control.

Every number the dashboard shows in live mode is computed here, by the real modules:

- core      `ResonantCausalWorldEngine` tracking the 16-node flow sandbox next to a frozen
            Transformer baseline. Predictions are scored *before* the engine sees the true
            next state; the world-line posterior and plastic update come afterwards.
- thermal   `Spectral2DMultimodalCodec` fusing a 16x16 thermal frame with 12 telemetry channels.
- rl        `SymplecticPontryaginAgent` stabilising a disturbed plant next to a frozen PD policy.
- swarm     two `DistributedWavefieldSwarm`s (gossip on / off) facing the same cascading fault.

`tick()` advances all four by one step and returns a JSON-ready snapshot.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

import torch
import torch.nn.functional as F

from rcwe.diagnostics.validation import ScientificValidationBenchmark
from rcwe.kernels.accelerator import describe_accelerator
from rcwe.multimodal.spectral_fusion import MultimodalBaselineTransformer, Spectral2DMultimodalCodec
from rcwe.rl.symplectic_rl import SymplecticPontryaginAgent
from rcwe.swarm.wavefield_swarm import DistributedWavefieldSwarm

__all__ = ["LiveMissionEngine", "ACTIONS", "REGIME_LABELS"]

REGIME_LABELS = {0: "W0 NOMINAL", 1: "W1 SINKHOLE FAULT", 2: "W2 TRUNK DEGRADATION"}
ACTIONS = ("scada_fault", "trunk_fault", "restore", "thermal_hotspot", "paint", "rl_shock", "swarm_cascade", "sleep", "step")
_EPISODE = 40            # sandbox steps before the plant is re-seeded (keeps it inside the trained envelope)
_POSTERIOR_RATE = 0.12   # smoothing of the world-line evidence: one step is too noisy to name a regime
_CONFIRM = 6             # consecutive steps a new leading world-line must hold before it is announced
_CASCADE_DELAY = 9       # ticks between the fault at substation 0 and its spread to substations 1-3
_CASCADE_LENGTH = 45


def _ema(previous: Optional[float], value: float, rate: float = 0.1) -> float:
    return value if previous is None else (1.0 - rate) * previous + rate * value


class LiveMissionEngine:
    """Thread-safe, stateful simulation that the Mission Control page polls."""

    def __init__(self, seed: int = 42, calibration_epochs: int = 40):
        self._lock = threading.Lock()
        started = time.perf_counter()
        torch.manual_seed(seed)

        # ---- core engine + baseline, calibrated on the three sandbox regimes --------------
        bench = ScientificValidationBenchmark(seed=seed)
        bench.train_and_calibrate(epochs=calibration_epochs)
        self.device = bench.device
        self.rcwe = bench.rcwe.eval()
        self.baseline = bench.baseline_tf.eval()
        self.sandbox = bench.sandbox
        self.rcwe.reset_runtime_adaptation()
        self.obs = self.sandbox.reset(batch_size=1, worldline_regime=0)
        self.field: Optional[torch.Tensor] = None
        self.regime = 0
        self._episode_step = 0
        self._identified = 0
        self._evidence: Optional[torch.Tensor] = None
        self._leader_run = 0

        # ---- track 1: multimodal thermal + telemetry -----------------------------------
        self._mm_channels, self._grid = 12, 16
        self.multimodal = Spectral2DMultimodalCodec(d_telemetry=self._mm_channels, grid_h=self._grid, grid_w=self._grid, d_model=64).to(self.device)
        self.mm_baseline = MultimodalBaselineTransformer(d_telemetry=self._mm_channels, grid_h=self._grid, grid_w=self._grid, d_model=64).to(self.device)
        axis = torch.linspace(-1.0, 1.0, self._grid, device=self.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self._spot = torch.exp(-3.0 * (xx ** 2 + yy ** 2))
        self._channels = torch.arange(self._mm_channels, device=self.device).float()
        self.heat = torch.zeros(self._grid, self._grid, device=self.device)
        self._train_multimodal()
        self._mm_alarm = False

        # ---- track 2: symplectic RL agent vs frozen PD policy ----------------------------
        self._rl_dim, self._rl_dt = 6, 0.08
        self.agent = SymplecticPontryaginAgent(d_state=self._rl_dim, d_action=self._rl_dim, d_hidden=48, dt=self._rl_dt).to(self.device)
        with torch.no_grad():
            self.agent.slow_actor.weight.zero_()
            for i in range(self._rl_dim):
                self.agent.slow_actor.weight[i, i] = -1.2
                self.agent.slow_actor.weight[i, self._rl_dim + i] = -0.8
        self.rl_q = torch.ones(1, self._rl_dim, device=self.device) * 0.35
        self.rl_p = torch.zeros(1, self._rl_dim, device=self.device)
        self.pd_q, self.pd_p = self.rl_q.clone(), self.rl_p.clone()
        self._rl_shock = 0

        # ---- track 3: wavefield swarm with and without gossip -----------------------------
        self._agents, self._node_dim = 4, 8
        self.swarm = DistributedWavefieldSwarm(num_agents=self._agents, d_node=self._node_dim, d_model=64, num_bands=16).to(self.device)
        self.swarm_isolated = DistributedWavefieldSwarm(num_agents=self._agents, d_node=self._node_dim, d_model=64, num_bands=16).to(self.device)
        self.swarm_isolated.load_state_dict(self.swarm.state_dict())
        self._cascade_at: Optional[int] = None

        # ---- bookkeeping ------------------------------------------------------------------
        self.step_count = 0
        self.last_action = "nominal"
        self._log: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._log_seq = 0
        self._kpi: Dict[str, Optional[float]] = {
            "rcwe_mse": None, "tf_mse": None, "ham_drift": None, "mm_rcwe": None, "mm_tf": None,
            "rl_dev": None, "pd_dev": None, "peak_da": 0.0, "cascade_coupled": None, "cascade_isolated": None,
            "precost_coupled": None,
        }
        self.scan_ms, self.scan_len = self._time_long_context()
        self.psi_kb = 0.0
        self.swarm_kb = float(self.swarm.psi_swarm.numel() * self.swarm.psi_swarm.element_size() / 1024.0)
        self._last: Dict[str, Any] = {}
        self._advance()                                           # so the first snapshot is populated
        self.log("OK", f"Engine calibrated in {time.perf_counter() - started:.1f}s on {describe_accelerator(self.device)}: "
                       f"{sum(p.numel() for p in self.rcwe.parameters()):,} RCWE parameters vs "
                       f"{sum(p.numel() for p in self.baseline.parameters()):,} in the Transformer baseline.")
        self.log("INFO", f"Standing wavefield is {self.psi_kb:.1f} KB and stays that size however long the stream runs. "
                         f"A {self.scan_len:,}-step scan takes {self.scan_ms:.1f} ms here.")

    # ------------------------------------------------------------------------------
    def log(self, level: str, message: str) -> None:
        self._log_seq += 1
        self._log.append({"seq": self._log_seq, "level": level, "msg": message, "step": self.step_count})

    def _time_long_context(self) -> Any:
        length = 32768
        x = torch.randn(1, length, 32, device=self.device)
        a = torch.randn(1, length, 8, device=self.device)
        try:
            with torch.no_grad():
                self.rcwe(x[:, :256], a[:, :256], plastic_runtime=False)
                begin = time.perf_counter()
                self.rcwe(x, a, plastic_runtime=False)
            return (time.perf_counter() - begin) * 1000.0, length
        except RuntimeError:
            return float("nan"), length

    # ---- track 1 helpers -----------------------------------------------------------------
    def _telemetry(self, t: int) -> torch.Tensor:
        return 0.5 * torch.sin(0.314 * t + 0.25 * self._channels)

    def _frame(self, t: int) -> torch.Tensor:
        return 0.25 * math.cos(0.314 * t) * self._spot + self.heat

    def _train_multimodal(self, epochs: int = 30, window: int = 24) -> None:
        steps = torch.arange(window + 1, device=self.device)
        telem = torch.stack([self._telemetry(int(t)) for t in steps]).unsqueeze(0)
        frames = torch.stack([self._frame(int(t)) for t in steps]).unsqueeze(0)
        x, y, f = telem[:, :-1], telem[:, 1:], frames[:, :-1]
        opt_r = torch.optim.AdamW(self.multimodal.parameters(), lr=4e-3)
        opt_t = torch.optim.AdamW(self.mm_baseline.parameters(), lr=4e-3)
        for _ in range(epochs):
            opt_r.zero_grad()
            F.mse_loss(self.multimodal(x, f, plastic_runtime=False)["y_hat"], y).backward()
            opt_r.step()
            opt_t.zero_grad()
            F.mse_loss(self.mm_baseline(x, f), y).backward()
            opt_t.step()
        self.multimodal.eval()
        self.mm_baseline.eval()

    # ---- regime control -------------------------------------------------------------------
    def _set_regime(self, regime: int) -> None:
        sb = self.sandbox
        sb.regime = regime
        sb.B = sb.base_B.clone()
        sb.sink_mask.zero_()
        if regime == 1:
            sb.sink_mask[5:7] = 0.45
        elif regime == 2:
            sb.B[4, 8] *= 0.35
            sb.B[8, 4] *= 0.35
        self.regime = regime

    # ------------------------------------------------------------------------------
    @torch.no_grad()
    def _advance(self) -> None:
        self.step_count += 1
        t = self.step_count
        dev = self.device

        # ---- core: predict, then observe, then adapt ------------------------------------
        self._episode_step += 1
        if self._episode_step > _EPISODE:
            regime = self.regime
            self.obs = self.sandbox.reset(batch_size=1, worldline_regime=regime)
            self._set_regime(regime)
            self._episode_step = 1
        action = torch.randn(1, 1, 8, device=dev) * 0.3
        x = self.obs.unsqueeze(1)
        next_obs, _ = self.sandbox.step(action.squeeze(1))
        y = next_obs.unsqueeze(1)
        prior = self.rcwe(x, action, standing_field=self.field, plastic_runtime=False)
        rcwe_mse = float(F.mse_loss(prior["output"], y))
        tf_mse = float(F.mse_loss(self.baseline(x, action), y))
        post = self.rcwe(x, action, standing_field=self.field, plastic_runtime=True, y_target=y)
        self.field = post["standing_field"]
        self.obs = next_obs
        violations = post["worldline_violations"].mean(dim=(1, 2))
        self._evidence = violations if self._evidence is None else (
            (1.0 - _POSTERIOR_RATE) * self._evidence + _POSTERIOR_RATE * violations)
        posteriors = F.softmax(-self._evidence * 8.0, dim=0)
        ham_drift = float(prior["energy_violation"])
        fast_norm = float(self.rcwe.metabolic_synapse.fast_weight.norm())
        self.psi_kb = float(self.field.numel() * self.field.element_size() / 1024.0)
        band = self.field[0]
        psi_mag = band.abs().mean(dim=-1)
        psi_phase = torch.angle(band.mean(dim=-1))

        leader = int(torch.argmax(posteriors))
        self._leader_run = self._leader_run + 1 if (leader != self._identified and float(posteriors[leader]) > 0.6) else 0
        if self._leader_run >= _CONFIRM:
            self._identified, self._leader_run = leader, 0
            verdict = "matches" if leader == self.regime else "does NOT match"
            self.log("OK" if leader == self.regime else "WARN",
                     f"World-line posterior now favours {REGIME_LABELS[leader]} (P = {float(posteriors[leader]):.2f}); "
                     f"this {verdict} the sandbox's actual regime ({REGIME_LABELS[self.regime]}).")

        # ---- track 1: multimodal fusion --------------------------------------------------
        self.heat.mul_(0.94)
        peak = float(self.heat.max())
        stress = min(peak / 2.4, 1.0)
        telem, frame = self._telemetry(t), self._frame(t)
        target = self._telemetry(t + 1).clone()
        target[:6] -= 1.35 * stress
        target[6:] += 1.15 * stress
        mm_x, mm_f, mm_y = telem.view(1, 1, -1), frame.view(1, 1, self._grid, self._grid), target.view(1, 1, -1)
        mm_out = self.multimodal(mm_x, mm_f, plastic_runtime=True, y_target=mm_y)
        mm_rcwe = float(F.mse_loss(mm_out["y_hat"], mm_y))
        mm_tf = float(F.mse_loss(self.mm_baseline(mm_x, mm_f), mm_y))
        if not self._mm_alarm and peak >= 1.0:
            self._mm_alarm = True
            self.log("CRIT", f"Thermal hotspot active (peak {peak:.2f}): the fused thermal+telemetry model is adapting online "
                             f"(error {mm_rcwe:.4f} vs {mm_tf:.4f} for the frozen multimodal Transformer).")
        elif self._mm_alarm and peak < 0.3:
            self._mm_alarm = False
            self.log("OK", f"Thermal grid back to nominal (cross-modal coherence {float(mm_out['cross_modal_coherence']):.3f}).")

        # ---- track 2: symplectic RL vs frozen PD -----------------------------------------
        amplitude = 1.35 if self._rl_shock > 0 else 0.05
        disturbance = torch.sin(torch.arange(self._rl_dim, device=dev).float() * 0.5 + 0.2 * t).unsqueeze(0) * amplitude
        with torch.enable_grad():
            a_rl = self.agent.select_pontryagin_action(self.rl_q, self.rl_p, horizon=3).detach()
        a_pd = F.linear(torch.cat([self.pd_q, self.pd_p], dim=-1), self.agent.slow_actor.weight)
        dt = self._rl_dt
        rl_p_next = 0.92 * self.rl_p + dt * (-1.5 * self.rl_q + a_rl + disturbance)
        rl_q_next = self.rl_q + dt * rl_p_next
        pd_p_next = 0.92 * self.pd_p + dt * (-1.5 * self.pd_q + a_pd + disturbance)
        pd_q_next = self.pd_q + dt * pd_p_next
        dopamine = 0.0
        if self._rl_shock > 0:
            correction = -(rl_q_next * 1.8 + rl_p_next * 1.4) / dt
            dopamine = self.agent.dopaminergic_plastic_update(self.rl_q, self.rl_p, a_rl, rl_q_next, rl_p_next, correction)
            self._rl_shock -= 1
            if self._rl_shock == 0:
                self.log("OK", f"RL disturbance over: plant deviation {float((rl_q_next ** 2).mean() + (rl_p_next ** 2).mean()):.4f} "
                               f"under the plastic policy vs {float((pd_q_next ** 2).mean() + (pd_p_next ** 2).mean()):.4f} under the frozen one.")
        self.rl_q, self.rl_p, self.pd_q, self.pd_p = rl_q_next, rl_p_next, pd_q_next, pd_p_next
        rl_dev = float((self.rl_q ** 2).mean() + (self.rl_p ** 2).mean())
        pd_dev = float((self.pd_q ** 2).mean() + (self.pd_p ** 2).mean())

        # ---- track 3: swarm under a cascading fault --------------------------------------
        base = 0.3 * torch.sin(torch.linspace(0, 3.14, self._node_dim, device=dev) + 0.15 * t).unsqueeze(0).expand(self._agents, -1)
        offset = torch.zeros(self._agents, self._node_dim, device=dev)
        mask = torch.zeros(self._agents, device=dev)
        since = None if self._cascade_at is None else t - self._cascade_at
        if since is not None:
            if since >= _CASCADE_LENGTH:
                self._cascade_at, since = None, None
                # The swarm only learns from agents flagged as faulted, so it cannot unlearn by itself.
                self.swarm.fast_swarm_weights.zero_()
                self.swarm_isolated.fast_swarm_weights.zero_()
                self.log("OK", "Swarm fault cleared; the swarms' plastic weights were reset for the next run.")
            else:
                offset[0], mask[0] = 1.45, 1.0
                if since >= _CASCADE_DELAY:
                    offset[1:], mask[1:] = 1.45, 1.0
        truth = base + offset
        y_coupled = self.swarm.step_swarm(base, truth, mask, enable_swarm_gossip=True)
        y_isolated = self.swarm_isolated.step_swarm(base, truth, mask, enable_swarm_gossip=False)
        err_coupled = ((y_coupled - truth) ** 2).mean(dim=-1)
        err_isolated = ((y_isolated - truth) ** 2).mean(dim=-1)
        if since is not None and 0 < since < _CASCADE_DELAY:
            # Gossip pre-loads the neighbours with the fault before it reaches them: that costs accuracy now.
            self._kpi["precost_coupled"] = _ema(self._kpi.get("precost_coupled") if since > 1 else None, float(err_coupled[1:].mean()), 0.5)
        if since is not None and since in (_CASCADE_DELAY, _CASCADE_DELAY + 1):
            coupled, isolated = float(err_coupled[1:].mean()), float(err_isolated[1:].mean())
            if since == _CASCADE_DELAY:
                self._kpi["cascade_coupled"], self._kpi["cascade_isolated"] = coupled, isolated
            else:
                self._kpi["cascade_coupled"] = 0.5 * (self._kpi["cascade_coupled"] + coupled)
                self._kpi["cascade_isolated"] = 0.5 * (self._kpi["cascade_isolated"] + isolated)
                ratio = self._kpi["cascade_isolated"] / max(self._kpi["cascade_coupled"], 1e-9)
                self.log("WARN", f"Cascade reached substations 1-3. First-impact error: {self._kpi['cascade_coupled']:.4f} with wavefield "
                                 f"gossip vs {self._kpi['cascade_isolated']:.4f} isolated ({ratio:.1f}x lower). The price: while waiting, "
                                 f"the pre-loaded neighbours mispredicted by {float(self._kpi.get('precost_coupled') or 0.0):.4f}.")

        # ---- running KPIs -------------------------------------------------------------------
        k = self._kpi
        k["rcwe_mse"], k["tf_mse"] = _ema(k["rcwe_mse"], rcwe_mse), _ema(k["tf_mse"], tf_mse)
        k["ham_drift"] = _ema(k["ham_drift"], ham_drift)
        k["mm_rcwe"], k["mm_tf"] = _ema(k["mm_rcwe"], mm_rcwe), _ema(k["mm_tf"], mm_tf)
        k["rl_dev"], k["pd_dev"] = _ema(k["rl_dev"], rl_dev), _ema(k["pd_dev"], pd_dev)
        k["peak_da"] = max(float(k["peak_da"] or 0.0) * 0.995, dopamine)

        self._last = {
            "core": {"rcwe_mse": rcwe_mse, "tf_mse": tf_mse, "ham_drift": ham_drift, "fast_norm": fast_norm,
                     "posteriors": [float(v) for v in posteriors]},
            "psi": {"mag": [float(v) for v in psi_mag], "phase": [float(v) for v in psi_phase]},
            "thermal": {"grid": [[round(float(v), 3) for v in row] for row in frame], "coherence": float(mm_out["cross_modal_coherence"]),
                        "peak": peak, "rcwe_mse": mm_rcwe, "tf_mse": mm_tf},
            "rl": {"q": float(self.rl_q[0, 0]), "p": float(self.rl_p[0, 0]), "fq": float(self.pd_q[0, 0]), "fp": float(self.pd_p[0, 0]),
                   "da": dopamine, "atp": float(self.agent.atp), "dev": rl_dev, "frozen_dev": pd_dev, "shock": self._rl_shock > 0},
            "swarm": {"err": [float(v) for v in err_coupled], "iso_err": [float(v) for v in err_isolated],
                      "fault": [bool(v) for v in mask > 0], "coupled_mse": float(err_coupled.mean()),
                      "isolated_mse": float(err_isolated.mean())},
        }

    # ------------------------------------------------------------------------------
    def tick(self, since_log: int = 0) -> Dict[str, Any]:
        """Advance one step and return the dashboard snapshot (log lines newer than `since_log`)."""
        with self._lock:
            self._advance()
            return self._snapshot(since_log)

    def snapshot(self, since_log: int = 0) -> Dict[str, Any]:
        """Current state without advancing."""
        with self._lock:
            return self._snapshot(since_log)

    def _snapshot(self, since_log: int) -> Dict[str, Any]:
        from rcwe import __version__

        def clean(value: Any) -> Any:
            if isinstance(value, float):
                return float(f"{value:.6g}") if math.isfinite(value) else None
            if isinstance(value, dict):
                return {key: clean(v) for key, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value

        payload = {
            "live": True, "version": __version__, "accelerator": describe_accelerator(self.device),
            "step": self.step_count, "last_action": self.last_action,
            "regime": self.regime, "regime_label": REGIME_LABELS[self.regime],
            "psi_kb": self.psi_kb, "swarm_kb": self.swarm_kb, "atp": self.rcwe.mean_atp,
            "scan_ms": self.scan_ms, "scan_len": self.scan_len,
            "kpi": dict(self._kpi),
            "log": [entry for entry in self._log if entry["seq"] > since_log],
        }
        payload.update(self._last)
        return clean(payload)

    def act(self, action: str, row: Optional[int] = None, col: Optional[int] = None) -> Dict[str, Any]:
        """Apply an operator action. Unknown actions are rejected rather than silently counted."""
        with self._lock:
            if action not in ACTIONS:
                return {"status": "error", "action": action, "error": f"unknown action; choose from {', '.join(ACTIONS)}"}
            self.last_action = action
            result: Dict[str, Any] = {"status": "ok", "action": action}
            if action == "scada_fault":
                self._set_regime(1)
                self.log("CRIT", "Topology mutation applied to the sandbox: sinkhole on relay nodes 5 and 6 (regime W1).")
            elif action == "trunk_fault":
                self._set_regime(2)
                self.log("WARN", "Topology mutation applied to the sandbox: trunk link 4-8 degraded to 35% susceptance (regime W2).")
            elif action == "restore":
                self._set_regime(0)
                self.heat.zero_()
                self.agent.fast_actor.zero_()
                self.agent.eligibility_trace.zero_()
                self._rl_shock = 0
                self.log("OK", "Nominal sandbox topology restored, thermal grid cleared, RL policy adaptation reset.")
            elif action == "thermal_hotspot":
                self.heat.copy_(torch.maximum(self.heat, 2.58 * self._spot))
                self.log("WARN", "Thermal hotspot injected (peak 2.58): telemetry channels sag in step with it.")
            elif action == "paint":
                r = max(0, min(self._grid - 1, int(row if row is not None else self._grid // 2)))
                c = max(0, min(self._grid - 1, int(col if col is not None else self._grid // 2)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self._grid and 0 <= cc < self._grid:
                            self.heat[rr, cc] = min(3.2, float(self.heat[rr, cc]) + (1.8 if dr == 0 and dc == 0 else 0.7))
                result.update(row=r, col=c)
            elif action == "rl_shock":
                self._rl_shock = 40
                self.rl_p += 1.45
                self.pd_p += 1.45
                self.log("WARN", "RL plant kicked (+1.45 momentum) and put under a strong oscillating disturbance for 40 steps.")
            elif action == "swarm_cascade":
                self._cascade_at = self.step_count + 1
                self.swarm.fast_swarm_weights.zero_()
                self.swarm_isolated.fast_swarm_weights.zero_()
                self.log("WARN", f"Fault injected at substation 0; it spreads to substations 1-3 in {_CASCADE_DELAY} steps.")
            elif action == "sleep":
                norm = self.rcwe.sleep_consolidate(consolidation_rate=0.45)
                result["consolidated_norm"] = round(norm, 5)
                self.log("OK", f"Sleep consolidation: folded fast weights (norm {norm:.4f}) into the slow weights and cleared them.")
            elif action == "step":
                self._advance()
            result["step"] = self.step_count
            return result
