import json
import math
import torch
import torch.nn.functional as F
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional

from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.sandbox.baseline_tf import BaselineCausalTransformer
from rcwe.sandbox.cpfn_env import CyberPhysicalFlowSandbox
from rcwe.kernels.accelerator import select_best_device, describe_accelerator
from rcwe.paths import resolve_output


class LiveRCWEController:
    """
    Stateful real-time simulation & control loop for the v0.6.0 Interactive Server.
    Supports live step streaming, manual anomaly injection (`regime = 0, 1, 2`),
    Sleep Consolidation (`consolidate_sleep_phase()`), and checkpoint snapshots.
    """
    def __init__(self):
        self.device = select_best_device()
        torch.manual_seed(42)
        self.sandbox = CyberPhysicalFlowSandbox(num_nodes=16, dt=0.05, device=self.device)
        self.rcwe = ResonantCausalWorldEngine(d_input=32, d_model=64, d_action=8, num_worldlines=3).to(self.device)
        self.tf_model = BaselineCausalTransformer(d_input=32, d_model=64, d_action=8, nhead=4, num_layers=2).to(self.device)
        self.standing_field: Optional[torch.Tensor] = None
        self.obs = self.sandbox.reset(batch_size=2, worldline_regime=0)
        self.step_count = 0
        self.anomaly_bias = torch.zeros(2, 1, 32, device=self.device)
        self._quick_warmup()

    def _quick_warmup(self) -> None:
        opt_r = torch.optim.AdamW(self.rcwe.parameters(), lr=4e-3)
        opt_t = torch.optim.AdamW(self.tf_model.parameters(), lr=4e-3)
        self.rcwe.train()
        self.tf_model.train()
        for _ in range(25):
            opt_r.zero_grad()
            opt_t.zero_grad()
            x = torch.randn(4, 6, 32, device=self.device) * 0.25
            a = torch.randn(4, 6, 8, device=self.device) * 0.25
            y = x + 0.05 * torch.sin(x)
            out_r = self.rcwe(x, a, plastic_runtime=False, y_target=y)
            (F.mse_loss(out_r["output"], y) + F.mse_loss(out_r["branch_predictions"][0], y)).backward()
            opt_r.step()
            F.mse_loss(self.tf_model(x, a), y).backward()
            opt_t.step()
        self.rcwe.eval()
        self.tf_model.eval()
        self.rcwe.reset_runtime_adaptation()

    @torch.no_grad()
    def step_tick(self) -> Dict[str, Any]:
        self.step_count += 1
        act = torch.randn(2, 1, 8, device=self.device) * 0.25
        x_curr = self.obs.unsqueeze(1)
        next_obs, _ = self.sandbox.step(act.squeeze(1))
        y_true = next_obs.unsqueeze(1) + self.anomaly_bias

        res_r = self.rcwe(x_curr, act, standing_field=self.standing_field, plastic_runtime=True, y_target=y_true)
        self.standing_field = res_r["standing_field"]
        res_t = self.tf_model(x_curr, act)
        self.obs = next_obs

        r_mse = float(F.mse_loss(res_r["output"], y_true).item())
        t_mse = float(F.mse_loss(res_t, y_true).item())
        atp = float(res_r["mean_atp"].item())
        fast_norm = float(self.rcwe.metabolic_synapse.fast_weight.abs().mean().item())
        psi_norm = float(self.standing_field.abs().mean().item())

        return {
            "step": self.step_count,
            "regime": self.sandbox.regime,
            "rcwe_mse": round(r_mse, 6),
            "tf_mse": round(t_mse, 6),
            "mean_atp": round(atp, 4),
            "fast_weight_norm": round(fast_norm, 5),
            "psi_coherence": round(psi_norm, 4),
            "accelerator": describe_accelerator(self.device),
        }

    def inject_anomaly(self, regime: int = 1) -> Dict[str, Any]:
        self.sandbox.regime = regime
        if regime == 1:
            self.anomaly_bias[:, :, :16] = 0.40
        elif regime == 2:
            self.anomaly_bias[:, :, :8] = -0.35
        else:
            self.anomaly_bias.zero_()
        return {"status": "anomaly_updated", "regime": regime}

    def trigger_sleep_consolidation(self) -> Dict[str, Any]:
        before_fast = float(self.rcwe.metabolic_synapse.fast_weight.abs().sum().item())
        self.rcwe.metabolic_synapse.consolidate_sleep_phase(consolidation_rate=0.1)
        after_fast = float(self.rcwe.metabolic_synapse.fast_weight.abs().sum().item())
        return {"status": "sleep_consolidated", "before_fast_norm": before_fast, "after_fast_norm": after_fast}

    def save_live_checkpoint(self, path: Optional[str] = None) -> Dict[str, Any]:
        path = resolve_output(path, "rcwe_live_server_ckpt.pt")
        saved = self.rcwe.save_pretrained(path, standing_field=self.standing_field)
        return {"status": "checkpoint_saved", "path": saved}


LIVE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Doradus Labs — RCWE v0.6.0 Live Interactive Control Server</title>
<style>
  body { background:#0b0f19; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,sans-serif; padding:24px; }
  .bar { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #23304d; padding-bottom:14px; margin-bottom:20px; }
  .btns { display:flex; gap:10px; }
  button { background:#1e293b; color:#fff; border:1px solid #38bdf8; padding:8px 14px; border-radius:8px; cursor:pointer; font-weight:600; }
  button:hover { background:#38bdf8; color:#0b0f19; }
  .grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:20px; }
  .card { background:#131b2e; border:1px solid #23304d; border-radius:10px; padding:16px; }
  .val { font-size:24px; font-weight:700; color:#38bdf8; margin-top:4px; }
  svg { width:100%; height:260px; background:#090d16; border:1px solid #23304d; border-radius:8px; }
</style>
</head>
<body>
  <div class="bar">
    <div>
      <h2 style="margin:0;">Doradus Labs — RCWE v0.6.0 Live Interactive Control Center</h2>
      <div style="font-size:13px; color:#94a3b8;" id="sub">Streaming Live Telemetry Ticks on Apple Silicon MPS</div>
    </div>
    <div class="btns">
      <button onclick="postAct('/api/anomaly?regime=1')">⚡ Inject Sinkhole Anomaly (W1)</button>
      <button onclick="postAct('/api/anomaly?regime=0')">✅ Restore Normal Regime (W0)</button>
      <button onclick="postAct('/api/sleep')">🌙 Trigger Sleep Consolidation</button>
      <button onclick="postAct('/api/save')">💾 Save Checkpoint (.pt)</button>
    </div>
  </div>
  <div class="grid">
    <div class="card"><div>Live Step / Regime</div><div class="val" id="v-step">--</div></div>
    <div class="card"><div>RCWE Live MSE</div><div class="val" style="color:#10b981;" id="v-rcwe">--</div></div>
    <div class="card"><div>Frozen Transformer MSE</div><div class="val" style="color:#f43f5e;" id="v-tf">--</div></div>
    <div class="card"><div>ATP Budget / |Psi|</div><div class="val" id="v-atp">--</div></div>
  </div>
  <div class="card">
    <h3 style="margin-top:0;">Real-Time Streaming Adaptation Curve (Green = RCWE, Red = Frozen Transformer)</h3>
    <svg id="chart" viewBox="0 0 800 240"></svg>
    <div id="status-log" style="margin-top:10px; font-size:12px; color:#38bdf8;">Ready.</div>
  </div>
<script>
  let rHist = [], tHist = [];
  async function tick() {
    const res = await fetch('/api/step');
    const d = await res.json();
    document.getElementById('v-step').textContent = `Step ${d.step} (W${d.regime})`;
    document.getElementById('v-rcwe').textContent = d.rcwe_mse.toFixed(5);
    document.getElementById('v-tf').textContent = d.tf_mse.toFixed(5);
    document.getElementById('v-atp').textContent = `${d.mean_atp.toFixed(2)}J | ${d.psi_coherence.toFixed(3)}`;
    rHist.push(d.rcwe_mse); tHist.push(d.tf_mse);
    if (rHist.length > 60) { rHist.shift(); tHist.shift(); }
    draw();
  }
  function draw() {
    const maxV = Math.max(...rHist, ...tHist, 0.05) * 1.1;
    const W = 800, H = 240, pad = 24;
    const pts = arr => arr.map((v, i) => `${pad + (i / Math.max(arr.length - 1, 1)) * (W - 2*pad)},${H - pad - (v / maxV) * (H - 2*pad)}`).join(' ');
    document.getElementById('chart').innerHTML =
      `<polyline fill="none" stroke="#f43f5e" stroke-width="2.5" points="${pts(tHist)}"/>` +
      `<polyline fill="none" stroke="#10b981" stroke-width="2.5" points="${pts(rHist)}"/>`;
  }
  async function postAct(url) {
    const res = await fetch(url, {method: 'POST'});
    const d = await res.json();
    document.getElementById('status-log').textContent = 'Action Executed: ' + JSON.stringify(d);
  }
  setInterval(tick, 400);
</script>
</body>
</html>
"""


def run_live_server(port: int = 8765) -> None:
    controller = LiveRCWEController()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/api/step"):
                payload = controller.step_tick()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(LIVE_HTML.encode("utf-8"))

        def do_POST(self):
            if self.path.startswith("/api/anomaly"):
                reg = 1 if "regime=1" in self.path else 0
                payload = controller.inject_anomaly(regime=reg)
            elif self.path.startswith("/api/sleep"):
                payload = controller.trigger_sleep_consolidation()
            elif self.path.startswith("/api/save"):
                payload = controller.save_live_checkpoint()
            else:
                payload = {"status": "unknown"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"RCWE v0.6.0 Live Interactive Server running at http://127.0.0.1:{port} (Press Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    run_live_server()
