"""
Mission Control: a live, six-panel view of the engine at work.

`rcwe ui` serves this page next to a `LiveMissionEngine` (see `rcwe.ui.live_engine`). In
that mode every curve and figure is computed by the engine modules on each tick, and the
buttons really act on the sandbox (topology faults, a thermal hotspot, an RL disturbance,
a cascading swarm fault, sleep consolidation). Dropping a telemetry file on the page runs
the toolkit's real analysis and links to the full report.

Opened on its own (for example the file written by `export_standalone_html`), the page has
no engine to talk to. It then runs a scripted animation and says so in a banner: nothing
it shows in that mode is a measurement.
"""
from pathlib import Path
from typing import Any, Dict, Optional

from rcwe.paths import resolve_output
from rcwe.ui.live_engine import LiveMissionEngine

__all__ = ["HTML_TEMPLATE", "MissionControlEngine", "serve_mission_control"]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Doradus Labs — RCWE Mission Control</title>
<style>
  :root {
    --bg-0: #060911;
    --bg-1: #0b111e;
    --bg-2: #111a2e;
    --border: rgba(56, 189, 248, 0.18);
    --border-bright: rgba(56, 189, 248, 0.45);
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
    --cyan: #38bdf8;
    --emerald: #10b981;
    --amber: #f59e0b;
    --rose: #f43f5e;
    --violet: #a855f7;
    --font-sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    --font-mono: "JetBrains Mono", "SF Mono", "Fira Code", Menlo, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: radial-gradient(circle at 15% 0%, #0f1b33 0%, var(--bg-0) 65%);
    color: var(--text-main);
    font-family: var(--font-sans);
    min-height: 100vh;
    overflow-x: hidden;
  }
  /* Topbar */
  .topbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: rgba(11, 17, 30, 0.88);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
    gap: 12px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-orb {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: conic-gradient(from 45deg, var(--cyan), var(--violet), var(--emerald), var(--cyan));
    box-shadow: 0 0 18px rgba(56, 189, 248, 0.55);
    position: relative;
  }
  .brand-orb::after {
    content: "";
    position: absolute;
    inset: 4px;
    border-radius: 50%;
    background: var(--bg-0);
  }
  .brand-title {
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.4px;
    color: #fff;
  }
  .brand-sub {
    font-size: 11px;
    color: var(--cyan);
    font-family: var(--font-mono);
  }
  .telemetry-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }
  .pill {
    background: rgba(17, 26, 46, 0.85);
    border: 1px solid var(--border);
    padding: 5px 10px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 11px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .pill .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
  }
  .pill .val { color: #fff; font-weight: 600; }

  /* Command Deck */
  .command-deck {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 12px 20px;
    background: rgba(8, 13, 24, 0.75);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    align-items: center;
    justify-content: space-between;
  }
  .btn-group {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }
  button.cmd-btn {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    color: var(--text-main);
    border: 1px solid var(--border);
    padding: 7px 12px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.16s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  button.cmd-btn:hover {
    border-color: var(--cyan);
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.25);
    transform: translateY(-1px);
  }
  button.cmd-btn.primary {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(6, 95, 70, 0.4));
    border-color: var(--emerald);
    color: #a7f3d0;
  }
  button.cmd-btn.danger {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.22), rgba(136, 19, 55, 0.4));
    border-color: var(--rose);
    color: #fecdd3;
  }
  button.cmd-btn.warn {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.22), rgba(120, 53, 15, 0.4));
    border-color: var(--amber);
    color: #fde68a;
  }
  button.cmd-btn.violet {
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.22), rgba(88, 28, 135, 0.4));
    border-color: var(--violet);
    color: #e9d5ff;
  }

  /* KPI Strip */
  .kpi-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 12px;
    padding: 14px 20px 4px 20px;
  }
  .kpi-card {
    background: linear-gradient(180deg, rgba(17, 26, 46, 0.8), rgba(11, 17, 30, 0.9));
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    position: relative;
    overflow: hidden;
  }
  .kpi-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-muted);
    margin-bottom: 4px;
  }
  .kpi-value {
    font-size: 21px;
    font-weight: 700;
    font-family: var(--font-mono);
    color: #fff;
  }
  .kpi-delta {
    font-size: 11px;
    font-family: var(--font-mono);
    margin-top: 4px;
    color: var(--emerald);
  }

  /* Main Bento Grid */
  .bento-grid {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 14px;
    padding: 14px 20px 24px 20px;
  }
  .panel {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.86), rgba(9, 14, 26, 0.94));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
  }
  .span-8 { grid-column: span 8; }
  .span-6 { grid-column: span 6; }
  .span-4 { grid-column: span 4; }
  .span-12 { grid-column: span 12; }
  @media (max-width: 1150px) {
    .span-8, .span-6, .span-4 { grid-column: span 12; }
  }
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 8px;
  }
  .panel-title {
    font-size: 13px;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-badge {
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 4px;
    background: rgba(56, 189, 248, 0.14);
    color: var(--cyan);
    border: 1px solid rgba(56, 189, 248, 0.3);
  }
  canvas {
    width: 100%;
    background: rgba(6, 9, 17, 0.75);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    display: block;
  }
  .legend-row {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-muted);
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }
  /* World-Line Posterior Bars */
  .wl-bars {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 4px;
  }
  .wl-row {
    display: grid;
    grid-template-columns: 150px 1fr 56px;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    font-family: var(--font-mono);
  }
  .wl-track {
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 4px;
    overflow: hidden;
  }
  .wl-fill {
    height: 100%;
    transition: width 0.2s ease;
  }
  /* Terminal / Diagnostic Log */
  .diag-console {
    background: #04070e;
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 8px;
    padding: 10px 12px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    line-height: 1.55;
    height: 190px;
    overflow-y: auto;
    color: #cbd5e1;
  }
  .log-line { margin-bottom: 4px; }
  .log-ts { color: #64748b; margin-right: 6px; }
  .log-crit { color: var(--rose); font-weight: 600; }
  .log-ok { color: var(--emerald); }
  .log-warn { color: var(--amber); }
  .log-info { color: var(--cyan); }

  /* Dropzone */
  .dropzone {
    border: 1px dashed var(--border-bright);
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    font-size: 12px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s ease;
    background: rgba(56, 189, 248, 0.04);
  }
  .dropzone:hover {
    background: rgba(56, 189, 248, 0.1);
    color: #fff;
  }
  .sim-banner {
    margin: 0; padding: 10px 20px; font-size: 13px; color: #fde68a;
    background: rgba(245, 158, 11, 0.14); border-bottom: 1px solid rgba(245, 158, 11, 0.45);
  }
  .sim-banner code { font-family: var(--font-mono); color: #fff; }
  .diag-console a { text-decoration: underline; }
</style>
</head>
<body>

<!-- TOPBAR -->
<header class="topbar">
  <div class="brand">
    <div class="brand-orb"></div>
    <div>
      <div class="brand-title">DORADUS LABS — RCWE MISSION CONTROL</div>
      <div class="brand-sub">Symplectic · Holographic O(1) Wavefield · 3-Factor Metabolic · Multimodal · Swarm</div>
    </div>
  </div>
  <div class="telemetry-pills">
    <div class="pill"><span class="dot" id="connDot"></span><span id="connLabel">CONNECTING…</span></div>
    <div class="pill">Step: <span class="val" id="stepVal">0</span></div>
    <div class="pill">|Ψ| State: <span class="val" id="psiKb" style="color:var(--cyan)">–</span></div>
    <div class="pill">Swarm |Ψ|: <span class="val" id="swarmKb" style="color:var(--violet)">–</span></div>
    <div class="pill">ATP Budget: <span class="val" id="atpVal" style="color:var(--emerald)">–</span></div>
    <div class="pill">Regime: <span class="val" id="regimeVal" style="color:var(--emerald)">–</span></div>
  </div>
</header>

<div class="sim-banner" id="simBanner" hidden>
  <strong>Simulation.</strong> This page is not connected to an engine, so the animation is scripted and nothing shown is a measurement.
  Start the live dashboard with <code>rcwe ui</code>.
</div>

<!-- INTERACTIVE COMMAND DECK -->
<section class="command-deck">
  <div class="btn-group">
    <button class="cmd-btn primary" id="playPauseBtn" onclick="togglePlay()">⏸ Pause Stream</button>
    <button class="cmd-btn" onclick="singleStep()">⏭ Single Step</button>
    <button class="cmd-btn danger" onclick="injectScadaFault()">⚡ Inject SCADA Fault (W₁)</button>
    <button class="cmd-btn warn" onclick="injectTrunkFault()">⚠ Degrade Trunk Link (W₂)</button>
    <button class="cmd-btn warn" onclick="injectThermalHotspot()">🔥 Inject 2D Thermal Hotspot</button>
    <button class="cmd-btn violet" onclick="injectRlDisturbance()">🎯 Symplectic RL Shock</button>
    <button class="cmd-btn warn" onclick="injectSwarmCascade()">🌐 Cascade Swarm Fault (Sub 0→3)</button>
    <button class="cmd-btn primary" onclick="triggerSleepConsolidation()">💤 Sleep Consolidate (W_fast→W_slow)</button>
    <button class="cmd-btn" onclick="restoreNominal()">↺ Restore Nominal</button>
  </div>
  <div class="btn-group">
    <label class="dropzone" for="csvFileInput">
      📂 Analyse your own CSV / JSON telemetry
      <input type="file" id="csvFileInput" accept=".csv,.tsv,.txt,.json,.jsonl" style="display:none" onchange="handleFileUpload(event)" />
    </label>
  </div>
</section>

<!-- KPI STRIP -->
<section class="kpi-strip">
  <div class="kpi-card">
    <div class="kpi-label">Multimodal 2D + telemetry error</div>
    <div class="kpi-value" id="kpiMmMse">–</div>
    <div class="kpi-delta" id="kpiMmGain">–</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">RCWE telemetry error (MSE)</div>
    <div class="kpi-value" id="kpiLiveMse">–</div>
    <div class="kpi-delta" id="kpiLiveGain">–</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Hamiltonian drift |ΔH|</div>
    <div class="kpi-value" id="kpiHamDrift">–</div>
    <div class="kpi-delta" id="kpiHamNote">–</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">RL plant deviation (plastic policy)</div>
    <div class="kpi-value" id="kpiRlDev">–</div>
    <div class="kpi-delta" id="kpiRlGain">–</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Swarm first-impact error</div>
    <div class="kpi-value" id="kpiSwarmMse">–</div>
    <div class="kpi-delta" id="kpiSwarmGain">–</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Long-context scan</div>
    <div class="kpi-value" id="kpiScan">–</div>
    <div class="kpi-delta" id="kpiScanNote">–</div>
  </div>
</section>

<!-- MAIN BENTO GRID -->
<main class="bento-grid">

  <!-- PANEL 1: Live Oscilloscope (RCWE vs Transformer MSE + Hamiltonian Energy) -->
  <div class="panel span-8">
    <div class="panel-header">
      <div class="panel-title">📈 Real-Time Telemetry Prediction Error & Symplectic Energy Conservation</div>
      <span class="panel-badge">LIVE STREAM</span>
    </div>
    <canvas id="oscCanvas" width="820" height="230" role="img" aria-label="Prediction error of RCWE and the frozen Transformer over time, with Hamiltonian drift and fast-weight norm"></canvas>
    <div class="legend-row">
      <div class="legend-item"><span class="swatch" style="background:#38bdf8"></span>RCWE error (scored before it sees the answer)</div>
      <div class="legend-item"><span class="swatch" style="background:#f43f5e"></span>Frozen Transformer error</div>
      <div class="legend-item"><span class="swatch" style="background:#10b981"></span>Hamiltonian drift |ΔH| (own scale)</div>
      <div class="legend-item"><span class="swatch" style="background:#f59e0b"></span>Fast-weight norm ‖W_fast‖ (own scale)</div>
    </div>
  </div>

  <!-- PANEL 2: 2D Thermal Infrared Array (16x16 Interactive Click/Drag) -->
  <div class="panel span-4">
    <div class="panel-header">
      <div class="panel-title">🔥 Track 1: 2D Spectral Thermal Array (16×16)</div>
      <span class="panel-badge">CLICK GRID TO INJECT HEAT</span>
    </div>
    <canvas id="thermalCanvas" width="380" height="200" style="cursor:crosshair"></canvas>
    <div class="legend-row" style="justify-content:space-between">
      <span>rfft2 Cross-Modal Coherence: <strong id="coherenceVal" style="color:var(--cyan)">–</strong></span>
      <span>Peak heat: <strong id="hotspotVal" style="color:var(--amber)">–</strong></span>
    </div>
  </div>

  <!-- PANEL 3: Holographic Wavefield |Psi| Phasor & 16-Band Spectrum -->
  <div class="panel span-4">
    <div class="panel-header">
      <div class="panel-title">🌀 O(1) Holographic Wavefield |Ψ| (16 Bands)</div>
      <span class="panel-badge">CONSTANT-SIZE STATE</span>
    </div>
    <canvas id="psiCanvas" width="380" height="210"></canvas>
    <div class="wl-bars">
      <div class="wl-row">
        <span>P(W₀: Nominal)</span>
        <div class="wl-track"><div class="wl-fill" id="barW0" style="width:0%;background:var(--emerald)"></div></div>
        <span id="txtW0">–</span>
      </div>
      <div class="wl-row">
        <span>P(W₁: Sinkhole Fault)</span>
        <div class="wl-track"><div class="wl-fill" id="barW1" style="width:0%;background:var(--rose)"></div></div>
        <span id="txtW1">–</span>
      </div>
      <div class="wl-row">
        <span>P(W₂: Trunk Degradation)</span>
        <div class="wl-track"><div class="wl-fill" id="barW2" style="width:0%;background:var(--amber)"></div></div>
        <span id="txtW2">–</span>
      </div>
    </div>
  </div>

  <!-- PANEL 4: Track 2 Symplectic Phase-Space (q, p) & H-Step RL Rollouts -->
  <div class="panel span-4">
    <div class="panel-header">
      <div class="panel-title">🎯 Track 2: Symplectic RL Phase-Space (q, p)</div>
      <span class="panel-badge">PONTRYAGIN + DOPAMINE TD</span>
    </div>
    <canvas id="rlCanvas" width="380" height="210"></canvas>
    <div class="legend-row" style="justify-content:space-between">
      <span>Dopamine δ_TD: <strong id="daVal" style="color:var(--violet)">–</strong></span>
      <span>Deviation: <strong id="pontVal" style="color:var(--emerald)">–</strong></span>
    </div>
  </div>

  <!-- PANEL 5: Track 3 Distributed 4-Agent Wavefield Swarm Topology -->
  <div class="panel span-4">
    <div class="panel-header">
      <div class="panel-title">🌐 Track 3: 4-Substation Wavefield Swarm</div>
      <span class="panel-badge">LAPLACIAN 8KB GOSSIP</span>
    </div>
    <canvas id="swarmCanvas" width="380" height="210"></canvas>
    <div class="legend-row" style="justify-content:space-between">
      <span>Error with gossip: <strong id="swCoupled" style="color:var(--emerald)">–</strong></span>
      <span>Isolated: <strong id="swIsolated" style="color:var(--rose)">–</strong></span>
      <span>Under each node: error with gossip / isolated</span>
    </div>
  </div>

  <!-- PANEL 6: Grounded Natural-Language Diagnostics & RCWE-Deep O(1) Code Stream -->
  <div class="panel span-12">
    <div class="panel-header">
      <div class="panel-title">🧠 Engine log: regime identification, interventions and file analyses</div>
      <span class="panel-badge">NEWEST FIRST</span>
    </div>
    <div class="diag-console" id="diagConsole"></div>
  </div>

</main>

<script>
// --- RCWE Mission Control -------------------------------------------------------------
// LIVE mode: every value comes from the engine through /api/v10/tick.
// SIMULATION mode (page opened as a file, or no engine reachable): curves are scripted so the
// layout can be previewed; the page says so and shows no measurements.
const HISTORY = 120;
const state = {
  live: false, playing: true, busy: false, step: 0, logSeq: 0, offline: 0,
  hist: { rcwe: [], tf: [], ham: [], fast: [] },
  probs: [1, 0, 0],
  thermal: Array.from({ length: 16 }, () => Array(16).fill(0)),
  psi: { mag: Array(16).fill(0), phase: Array(16).fill(0) },
  rlTrail: [], frozenTrail: [],
  swarm: { err: [0, 0, 0, 0], iso: [0, 0, 0, 0], fault: [false, false, false, false] },
  pulses: [],
  sim: { regime: 0, wFast: 0.04, atp: 1.0, rlQ: 0.35, rlP: 0, fQ: 0.35, fP: 0, fault: [0, 0, 0, 0] }
};
const $ = (id) => document.getElementById(id);

function num(v, digits) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && a < 1e-3) return v.toExponential(1);
  return v.toFixed(digits === undefined ? (a >= 100 ? 1 : a >= 1 ? 3 : 4) : digits);
}
function versus(ours, theirs, what) {
  if (!(ours > 0) || !(theirs > 0)) return what + ": –";
  const r = theirs / ours;
  return (r >= 1 ? r.toFixed(1) + "x lower than " : (1 / r).toFixed(1) + "x higher than ") + what + " (" + num(theirs) + ")";
}
function push(arr, v) { arr.push(v); if (arr.length > HISTORY) arr.shift(); }

function appendLog(level, msg, step) {
  const el = $("diagConsole");
  const cls = level === "CRIT" ? "log-crit" : level === "WARN" ? "log-warn" : level === "OK" ? "log-ok" : "log-info";
  const line = document.createElement("div");
  line.className = "log-line";
  const ts = document.createElement("span");
  ts.className = "log-ts";
  ts.textContent = "[" + new Date().toISOString().split("T")[1].slice(0, 8) + " | t=" + (step === undefined ? state.step : step) + "]";
  const tag = document.createElement("span");
  tag.className = cls;
  tag.textContent = "[" + level + "] ";
  line.appendChild(ts); line.appendChild(tag); line.appendChild(document.createTextNode(msg));
  el.prepend(line);
  while (el.children.length > 60) el.removeChild(el.lastChild);
  return line;
}

// ---------------------------------------------------------------- live mode
function applyServer(s) {
  if (s.step < state.step) {
    // The engine was restarted under this page: its step and log counters began again.
    state.logSeq = 0;
    Object.keys(state.hist).forEach((k) => { state.hist[k].length = 0; });
    state.rlTrail.length = 0; state.frozenTrail.length = 0;
    appendLog("WARN", "The engine was restarted; the charts start again from here.");
  }
  state.step = s.step;
  $("stepVal").textContent = s.step;
  $("atpVal").textContent = num(s.atp, 3);
  $("psiKb").textContent = num(s.psi_kb, 1) + " KB";
  $("swarmKb").textContent = num(s.swarm_kb, 1) + " KB";
  const regimeEl = $("regimeVal");
  regimeEl.textContent = s.regime_label;
  regimeEl.style.color = s.regime === 0 ? "var(--emerald)" : s.regime === 1 ? "var(--rose)" : "var(--amber)";

  push(state.hist.rcwe, s.core.rcwe_mse); push(state.hist.tf, s.core.tf_mse);
  push(state.hist.ham, s.core.ham_drift); push(state.hist.fast, s.core.fast_norm);
  state.probs = s.core.posteriors;
  state.thermal = s.thermal.grid;
  state.psi = s.psi;
  push(state.rlTrail, { q: s.rl.q, p: s.rl.p }); push(state.frozenTrail, { q: s.rl.fq, p: s.rl.fp });
  if (state.rlTrail.length > 60) { state.rlTrail.shift(); state.frozenTrail.shift(); }
  state.swarm = { err: s.swarm.err, iso: s.swarm.iso_err, fault: s.swarm.fault };

  const k = s.kpi;
  $("kpiMmMse").textContent = num(k.mm_rcwe);
  $("kpiMmGain").textContent = versus(k.mm_rcwe, k.mm_tf, "multimodal Transformer");
  $("kpiLiveMse").textContent = num(k.rcwe_mse);
  $("kpiLiveGain").textContent = versus(k.rcwe_mse, k.tf_mse, "frozen Transformer");
  $("kpiHamDrift").textContent = num(k.ham_drift);
  $("kpiHamNote").textContent = "energy change per symplectic step";
  $("kpiRlDev").textContent = num(k.rl_dev);
  $("kpiRlGain").textContent = "frozen PD policy: " + num(k.pd_dev) + " · peak δ_TD " + num(k.peak_da, 2);
  $("kpiSwarmMse").textContent = k.cascade_coupled === null ? "–" : num(k.cascade_coupled);
  $("kpiSwarmGain").textContent = k.cascade_coupled === null ? "run a cascade to measure first-impact error"
    : versus(k.cascade_coupled, k.cascade_isolated, "isolated agents") + " · pre-impact cost " + num(k.precost_coupled);
  $("kpiScan").textContent = s.scan_ms === null ? "–" : num(s.scan_ms, 1) + " ms";
  $("kpiScanNote").textContent = s.scan_len.toLocaleString() + " steps in one pass · state stays " + num(s.psi_kb, 1) + " KB";

  $("coherenceVal").textContent = num(s.thermal.coherence, 3);
  $("hotspotVal").textContent = num(s.thermal.peak, 2);
  $("daVal").textContent = num(s.rl.da, 2);
  $("pontVal").textContent = num(s.rl.dev) + " vs " + num(s.rl.frozen_dev) + " frozen";
  $("swCoupled").textContent = num(s.swarm.coupled_mse);
  $("swIsolated").textContent = num(s.swarm.isolated_mse);

  (s.log || []).forEach((entry) => { if (entry.seq > state.logSeq) { state.logSeq = entry.seq; appendLog(entry.level, entry.msg, entry.step); } });
  updateBars();
  renderAllCanvases(s.step * 0.08);
}

async function liveTick() {
  if (state.busy) return;
  state.busy = true;
  try {
    const res = await fetch("/api/v10/tick?since=" + state.logSeq, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    applyServer(await res.json());
    if (state.offline) { state.offline = 0; setConnection(true, state.accelerator); }
  } catch (err) {
    state.offline += 1;
    if (state.offline === 3) { setConnection(false, "ENGINE UNREACHABLE"); appendLog("CRIT", "Lost contact with the engine. Values are frozen until it is back."); }
  } finally { state.busy = false; }
}

async function sendAction(type, extra) {
  if (!state.live) return null;
  try {
    const res = await fetch("/api/v10/action?type=" + encodeURIComponent(type) + (extra || ""), { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    return await res.json();
  } catch (err) { appendLog("CRIT", "Action '" + type + "' could not reach the engine."); return null; }
}

function setConnection(ok, label) {
  $("connDot").style.background = ok ? "var(--emerald)" : "var(--rose)";
  $("connDot").style.boxShadow = "0 0 8px " + (ok ? "var(--emerald)" : "var(--rose)");
  $("connLabel").textContent = label;
}

// ---------------------------------------------------------------- operator actions
function togglePlay() {
  state.playing = !state.playing;
  $("playPauseBtn").textContent = state.playing ? "⏸ Pause Stream" : "▶ Resume Stream";
}
function singleStep() { if (state.live) liveTick(); else stepSimulation(); }
function injectScadaFault() {
  if (state.live) { sendAction("scada_fault"); return; }
  state.sim.regime = 1; state.probs = [0.04, 0.91, 0.05];
  appendLog("WARN", "[SIMULATED] sinkhole fault on nodes 5 and 6.");
}
function injectTrunkFault() {
  if (state.live) { sendAction("trunk_fault"); return; }
  state.sim.regime = 2; state.probs = [0.05, 0.07, 0.88];
  appendLog("WARN", "[SIMULATED] trunk link degraded.");
}
function restoreNominal() {
  if (state.live) { sendAction("restore"); return; }
  state.sim.regime = 0; state.probs = [0.92, 0.05, 0.03];
  state.thermal = Array.from({ length: 16 }, () => Array(16).fill(0));
  appendLog("OK", "[SIMULATED] nominal regime restored.");
}
function injectThermalHotspot() {
  if (state.live) { sendAction("thermal_hotspot"); return; }
  for (let r = 0; r < 16; r++) for (let c = 0; c < 16; c++) {
    const d = Math.hypot(r - 7.5, c - 7.5);
    state.thermal[r][c] = Math.max(state.thermal[r][c], 2.58 * Math.exp(-0.06 * d * d));
  }
  appendLog("WARN", "[SIMULATED] thermal hotspot.");
}
function injectRlDisturbance() {
  if (state.live) { sendAction("rl_shock"); return; }
  state.sim.rlP += 1.45; state.sim.fP += 1.45;
  appendLog("WARN", "[SIMULATED] RL plant kicked.");
}
function injectSwarmCascade() {
  for (let i = 0; i < 4; i++) state.pulses.push({ from: i, to: (i + 1) % 4, p: 0.0 });
  if (state.live) { sendAction("swarm_cascade"); return; }
  state.sim.fault = [1.45, 0.9, 0.6, 0.9];
  appendLog("WARN", "[SIMULATED] cascading fault at substation 0.");
}
function triggerSleepConsolidation() {
  if (state.live) { sendAction("sleep"); return; }
  state.sim.wFast *= 0.25;
  appendLog("OK", "[SIMULATED] sleep consolidation.");
}

// Click or drag on the thermal grid to add heat
const thermCanvas = $("thermalCanvas");
let paintingHeat = false, lastPaint = "";
function paintAt(e) {
  const rect = thermCanvas.getBoundingClientRect();
  const col = Math.floor((e.clientX - rect.left) / rect.width * 16), row = Math.floor((e.clientY - rect.top) / rect.height * 16);
  if (row < 0 || row > 15 || col < 0 || col > 15 || lastPaint === row + ":" + col) return;
  lastPaint = row + ":" + col;
  if (state.live) { sendAction("paint", "&r=" + row + "&c=" + col); return; }
  for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
    const nr = row + dr, nc = col + dc;
    if (nr >= 0 && nr < 16 && nc >= 0 && nc < 16) state.thermal[nr][nc] = Math.min(3.2, state.thermal[nr][nc] + (dr === 0 && dc === 0 ? 1.8 : 0.7));
  }
}
thermCanvas.addEventListener("mousedown", (e) => { paintingHeat = true; lastPaint = ""; paintAt(e); });
window.addEventListener("mouseup", () => { paintingHeat = false; });
thermCanvas.addEventListener("mousemove", (e) => { if (paintingHeat) paintAt(e); });

// Upload a telemetry file: analysed for real by the engine's toolkit (never in the browser)
function handleFileUpload(event) {
  const file = event.target.files[0];
  event.target.value = "";
  analyseFile(file);
}
// Dropping a file anywhere on the page does the same as choosing one.
window.addEventListener("dragover", (e) => { e.preventDefault(); });
window.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) analyseFile(e.dataTransfer.files[0]);
});
function analyseFile(file) {
  if (!file) return;
  if (!state.live) { appendLog("WARN", "Analysing a file needs the engine. Start it with `rcwe ui`, or run `rcwe analyze " + file.name + "`."); return; }
  if (file.size > 56 * 1024 * 1024) { appendLog("CRIT", file.name + " is larger than 56 MB; use `rcwe analyze` on the command line instead."); return; }
  const reader = new FileReader();
  reader.onload = async function (e) {
    appendLog("INFO", "Uploading " + file.name + " (" + Math.round(file.size / 1024) + " KB) for analysis…");
    try {
      const res = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: file.name, text: e.target.result, train_fraction: 0.6 }) });
      const job = await res.json();
      if (!res.ok) throw new Error(job.error || "upload rejected");
      pollJob(job.job, file.name);
    } catch (err) { appendLog("CRIT", "Could not analyse " + file.name + ": " + err.message); }
  };
  reader.readAsText(file);
}
async function pollJob(id, name) {
  try {
    const res = await fetch("/api/jobs/" + id, { cache: "no-store" });
    const job = await res.json();
    if (!res.ok) { appendLog("CRIT", name + ": the analysis is no longer available (" + (job.error || "HTTP " + res.status) + ")."); return; }
    if (job.status === "done") {
      const line = appendLog(job.events ? "CRIT" : "OK", name + ": " + job.headline + " ");
      const link = document.createElement("a");
      link.href = "/api/jobs/" + id + "/report"; link.target = "_blank"; link.rel = "noopener";
      link.textContent = "Open the full report"; link.style.color = "var(--cyan)";
      line.appendChild(link);
      (job.event_lines || []).slice(0, 4).forEach((text) => appendLog("WARN", text));
    } else if (job.status === "error") {
      appendLog("CRIT", name + ": " + job.error);
    } else { setTimeout(() => pollJob(id, name), 700); }
  } catch (err) { appendLog("CRIT", "Lost track of the analysis job for " + name + "."); }
}

// ---------------------------------------------------------------- simulation mode
function stepSimulation() {
  state.step += 1;
  const t = state.step * 0.08, sim = state.sim;
  let maxHeat = 0;
  for (let r = 0; r < 16; r++) for (let c = 0; c < 16; c++) {
    const wave = 0.22 * Math.sin(0.35 * r + 0.25 * c + t) * Math.cos(0.2 * r - 0.3 * t);
    state.thermal[r][c] = state.thermal[r][c] * 0.94 + Math.max(0, wave) * 0.06;
    maxHeat = Math.max(maxHeat, state.thermal[r][c]);
  }
  const fault = (sim.regime !== 0 || maxHeat > 1.2) ? 1 : 0;
  sim.wFast = fault ? Math.min(1.45, sim.wFast + 0.04) : sim.wFast * 0.995;
  push(state.hist.rcwe, 0.0007 + fault * 0.018 * Math.exp(-sim.wFast * 2.4) + 0.0003 * Math.abs(Math.sin(t)));
  push(state.hist.tf, 0.015 + fault * 0.38 + 0.006 * Math.abs(Math.cos(t * 0.7)));
  push(state.hist.ham, 0.011 + 0.003 * Math.abs(Math.sin(t * 0.5)));
  push(state.hist.fast, sim.wFast);
  for (let b = 0; b < 16; b++) { state.psi.mag[b] = 0.25 + 0.7 * Math.abs(Math.sin(t * 0.4 + b * 0.35)); state.psi.phase[b] = t * 0.08 * (b + 1) + b * 0.4; }
  const d = Math.sin(t * 1.3) * 0.04;
  sim.rlP = 0.92 * sim.rlP + 0.08 * (-3.15 * sim.rlQ - 1.15 * sim.rlP + d); sim.rlQ += 0.08 * sim.rlP;
  sim.fP = 0.94 * sim.fP + 0.08 * (-2.05 * sim.fQ - 0.25 * sim.fP + d * 2.5); sim.fQ += 0.08 * sim.fP;
  push(state.rlTrail, { q: sim.rlQ, p: sim.rlP }); push(state.frozenTrail, { q: sim.fQ, p: sim.fP });
  if (state.rlTrail.length > 60) { state.rlTrail.shift(); state.frozenTrail.shift(); }
  sim.fault = sim.fault.map((f) => f * 0.95);
  state.swarm = { err: sim.fault.map((f) => 0.015 + 0.06 * f), iso: sim.fault.map((f) => 0.015 + 0.4 * f), fault: sim.fault.map((f) => f > 0.25) };
  $("stepVal").textContent = state.step;
  updateBars();
  renderAllCanvases(t);
}
function enterSimulation() {
  state.live = false;
  $("simBanner").hidden = false;
  setConnection(false, "SIMULATION — NOT CONNECTED");
  ["kpiMmMse", "kpiLiveMse", "kpiHamDrift", "kpiRlDev", "kpiSwarmMse", "kpiScan"].forEach((id) => { $(id).textContent = "–"; });
  ["kpiMmGain", "kpiLiveGain", "kpiHamNote", "kpiRlGain", "kpiSwarmGain", "kpiScanNote"].forEach((id) => { $(id).textContent = "measured only when connected to an engine"; });
  ["atpVal", "psiKb", "swarmKb", "coherenceVal", "hotspotVal", "daVal", "pontVal", "swCoupled", "swIsolated"].forEach((id) => { $(id).textContent = "–"; });
  $("regimeVal").textContent = "SIMULATED";
  appendLog("WARN", "This page is not connected to an engine: the animation is scripted and no figure here is a measurement. Run `rcwe ui` for the live dashboard.");
}

// ---------------------------------------------------------------- drawing
function updateBars() {
  ["W0", "W1", "W2"].forEach((name, i) => {
    const p = Math.max(0, Math.min(1, state.probs[i] || 0)) * 100;
    $("bar" + name).style.width = p.toFixed(1) + "%";
    $("txt" + name).textContent = p.toFixed(1) + "%";
  });
}
function renderAllCanvases(t) { drawOscilloscope(); drawThermalGrid(); drawPsiWavefield(); drawRlPhaseSpace(); drawSwarmNetwork(t); }

function drawOscilloscope() {
  const c = $("oscCanvas"), ctx = c.getContext("2d"), W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const left = 58, right = 12, mainTop = 10, mainH = 128, stripH = 30, gap = 12;
  ctx.font = "10px monospace"; ctx.textBaseline = "middle";

  // Main plot: prediction error of both models on ONE shared log axis
  const both = state.hist.rcwe.concat(state.hist.tf).filter((v) => v > 0);
  let lo = -5, hi = 0;
  if (both.length) { lo = Math.floor(Math.log10(Math.min.apply(null, both))); hi = Math.ceil(Math.log10(Math.max.apply(null, both))); if (hi <= lo) hi = lo + 1; }
  const yLog = (v) => mainTop + (1 - (Math.log10(Math.max(v, Math.pow(10, lo))) - lo) / (hi - lo)) * mainH;
  for (let e = lo; e <= hi; e++) {
    const y = mainTop + (1 - (e - lo) / (hi - lo)) * mainH;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.10)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(W - right, y); ctx.stroke();
    ctx.fillStyle = "#94a3b8"; ctx.textAlign = "right"; ctx.fillText("1e" + e, left - 6, y);
  }
  ctx.save(); ctx.translate(11, mainTop + mainH / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillStyle = "#94a3b8"; ctx.fillText("MSE (log)", 0, 0); ctx.restore();
  const xAt = (i, n) => left + (n <= 1 ? 0 : i / (n - 1)) * (W - left - right);
  function line(arr, color, yOf, width) {
    if (arr.length < 2) return;
    ctx.beginPath();
    arr.forEach((v, i) => { const x = xAt(i, arr.length), y = yOf(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.strokeStyle = color; ctx.lineWidth = width || 2; ctx.lineJoin = "round"; ctx.stroke();
  }
  line(state.hist.tf, "#f43f5e", yLog); line(state.hist.rcwe, "#38bdf8", yLog);

  // Two strips underneath, each with its own scale and unit
  function strip(arr, color, top, label) {
    const max = Math.max.apply(null, arr.concat([1e-12])), min = Math.min.apply(null, arr.concat([0]));
    ctx.strokeStyle = "rgba(148, 163, 184, 0.14)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, top + stripH); ctx.lineTo(W - right, top + stripH); ctx.stroke();
    line(arr, color, (v) => top + stripH - (max > min ? (v - min) / (max - min) : 0) * (stripH - 4) - 1, 1.6);
    ctx.fillStyle = "#94a3b8"; ctx.textAlign = "right"; ctx.fillText(label, left - 6, top + stripH / 2);
    ctx.textAlign = "right"; ctx.fillStyle = "#e2e8f0"; ctx.fillText("now " + (arr.length ? num(arr[arr.length - 1]) : "–"), W - right - 2, top + 5);
  }
  strip(state.hist.ham, "#10b981", mainTop + mainH + gap, "|ΔH|");
  strip(state.hist.fast, "#f59e0b", mainTop + mainH + gap + stripH + gap - 4, "‖W_fast‖");
}

function drawThermalGrid() {
  const c = $("thermalCanvas"), ctx = c.getContext("2d"), W = c.width, H = c.height, cw = W / 16, ch = H / 16;
  ctx.clearRect(0, 0, W, H);
  for (let r = 0; r < 16; r++) for (let col = 0; col < 16; col++) {
    const v = Math.max(0, Math.min(1, (state.thermal[r][col] + 0.25) / 2.85));
    const red = Math.floor(20 + 235 * Math.pow(v, 0.7)), green = Math.floor(30 + 180 * Math.pow(v, 1.6) - 90 * (1 - v)), blue = Math.floor(90 * (1 - v) + 40 * (v > 0.8 ? 1 : 0));
    ctx.fillStyle = "rgb(" + Math.max(12, red) + ", " + Math.max(18, green) + ", " + Math.max(25, blue) + ")";
    ctx.fillRect(col * cw + 1, r * ch + 1, cw - 2, ch - 2);
  }
}

function drawPsiWavefield() {
  const c = $("psiCanvas"), ctx = c.getContext("2d"), W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const cx = 95, cy = H / 2, rad = 72;
  const mags = state.psi.mag.map((m) => Math.sqrt(Math.max(m, 0))), maxMag = Math.max.apply(null, mags.concat([1e-9]));
  ctx.strokeStyle = "rgba(56, 189, 248, 0.2)"; ctx.lineWidth = 1;
  [24, 48, 72].forEach((r) => { ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke(); });
  for (let b = 0; b < 16; b++) {            // one phasor per frequency band: length = |Psi_b|, angle = its mean phase
    const amp = mags[b] / maxMag * rad, px = cx + amp * Math.cos(state.psi.phase[b]), py = cy + amp * Math.sin(state.psi.phase[b]);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.strokeStyle = "hsla(" + (190 + b * 7) + ", 90%, 60%, 0.55)"; ctx.stroke();
    ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fillStyle = "hsl(" + (190 + b * 7) + ", 95%, 68%)"; ctx.fill();
  }
  const startX = 190, barW = 9, gapX = 2;
  for (let b = 0; b < 16; b++) {
    const bh = mags[b] / maxMag * (H - 36);
    ctx.fillStyle = "hsl(" + (190 + b * 6) + ", 88%, 58%)";
    ctx.fillRect(startX + b * (barW + gapX), H - 16 - bh, barW, bh);
  }
  ctx.fillStyle = "#94a3b8"; ctx.font = "10px monospace"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  ctx.fillText("band 1", startX, H - 3); ctx.textAlign = "right"; ctx.fillText("16", startX + 16 * (barW + gapX), H - 3);
  ctx.textAlign = "left"; ctx.fillText("|Ψ| per band (√ scale)", startX, 10);
}

function drawRlPhaseSpace() {
  const c = $("rlCanvas"), ctx = c.getContext("2d"), W = c.width, H = c.height, cx = W / 2, cy = H / 2;
  ctx.clearRect(0, 0, W, H);
  const pts = state.rlTrail.concat(state.frozenTrail);
  let qMax = 0.05, pMax = 0.05;
  pts.forEach((pt) => { qMax = Math.max(qMax, Math.abs(pt.q)); pMax = Math.max(pMax, Math.abs(pt.p)); });
  const sx = (W / 2 - 22) / qMax, sy = (H / 2 - 22) / pMax;
  ctx.strokeStyle = "rgba(148, 163, 184, 0.14)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
  [0.5, 1].forEach((f) => { ctx.beginPath(); ctx.ellipse(cx, cy, f * (W / 2 - 22), f * (H / 2 - 22), 0, 0, Math.PI * 2); ctx.stroke(); });
  function trail(list, color, width, dot) {
    if (list.length < 2) return;
    ctx.beginPath();
    list.forEach((pt, i) => { const x = cx + pt.q * sx, y = cy - pt.p * sy; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    const last = list[list.length - 1];
    ctx.beginPath(); ctx.arc(cx + last.q * sx, cy - last.p * sy, 4.5, 0, Math.PI * 2); ctx.fillStyle = dot; ctx.fill();
  }
  trail(state.frozenTrail, "rgba(244, 63, 94, 0.65)", 1.8, "#f43f5e");
  trail(state.rlTrail, "#10b981", 2.5, "#38bdf8");
  ctx.fillStyle = "#94a3b8"; ctx.font = "10px monospace"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  ctx.textAlign = "right"; ctx.fillText("q = ±" + num(qMax, 2), W - 4, cy - 5);
  ctx.textAlign = "left"; ctx.fillText("p = ±" + num(pMax, 2), cx + 6, 12);
  ctx.fillStyle = "#10b981"; ctx.fillText("plastic policy", 8, H - 20); ctx.fillStyle = "#f43f5e"; ctx.fillText("frozen PD policy", 8, H - 7);
}

function drawSwarmNetwork(t) {
  const c = $("swarmCanvas"), ctx = c.getContext("2d"), W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const coords = [{ x: W / 2, y: 30 }, { x: W - 84, y: H / 2 - 8 }, { x: W / 2, y: H - 46 }, { x: 84, y: H / 2 - 8 }];
  ctx.strokeStyle = "rgba(168, 85, 247, 0.32)"; ctx.lineWidth = 2;
  for (let i = 0; i < 4; i++) { const a = coords[i], b = coords[(i + 1) % 4]; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
  if (state.step % 16 === 0 && state.pulses.length < 8) state.pulses.push({ from: (state.step / 16) % 4, to: ((state.step / 16) + 1) % 4, p: 0 });
  for (let i = state.pulses.length - 1; i >= 0; i--) {
    const pulse = state.pulses[i]; pulse.p += state.live ? 0.12 : 0.045;
    if (pulse.p >= 1) { state.pulses.splice(i, 1); continue; }
    const a = coords[pulse.from], b = coords[pulse.to];
    ctx.beginPath(); ctx.arc(a.x + (b.x - a.x) * pulse.p, a.y + (b.y - a.y) * pulse.p, 4, 0, Math.PI * 2); ctx.fillStyle = "#38bdf8"; ctx.fill();
  }
  ctx.font = "bold 11px monospace"; ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
  coords.forEach((pt, i) => {
    const faulted = state.swarm.fault[i];
    ctx.beginPath(); ctx.arc(pt.x, pt.y, 18, 0, Math.PI * 2);
    ctx.fillStyle = faulted ? "rgba(244, 63, 94, 0.3)" : "rgba(16, 185, 129, 0.22)"; ctx.fill();
    ctx.strokeStyle = faulted ? "#f43f5e" : "#10b981"; ctx.lineWidth = 2.2; ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.fillText("S" + i, pt.x, pt.y + 4);
    ctx.font = "10px monospace"; ctx.fillStyle = "#94a3b8";
    ctx.fillText(num(state.swarm.err[i], 3) + " / " + num(state.swarm.iso[i], 3), pt.x, pt.y + 33);
    ctx.font = "bold 11px monospace";
  });
}

// ---------------------------------------------------------------- start-up
async function boot() {
  try {
    const res = await fetch("/api/v10/state", { cache: "no-store" });
    if (!res.ok) throw new Error("no engine");
    const snapshot = await res.json();
    if (!snapshot.live) throw new Error("no engine");
    state.live = true;
    state.accelerator = "LIVE · " + snapshot.accelerator + " · RCWE " + snapshot.version;
    setConnection(true, state.accelerator);
    applyServer(snapshot);
  } catch (err) { enterSimulation(); }
  setInterval(() => { if (!state.playing) return; if (state.live) liveTick(); else stepSimulation(); }, state.live ? 250 : 90);
}
renderAllCanvases(0);
boot();
</script>
</body>
</html>
"""


class MissionControlEngine(LiveMissionEngine):
    """Backend controller for the Mission Control page (the live engine plus page export)."""

    def trigger_action(self, action_type: str, **params: Any) -> Dict[str, Any]:
        """Apply an operator action: scada_fault, trunk_fault, restore, thermal_hotspot, paint,
        rl_shock, swarm_cascade, sleep or step."""
        return self.act(action_type, row=params.get("row"), col=params.get("col"))

    def get_snapshot(self) -> Dict[str, Any]:
        """Current dashboard state without advancing the simulation."""
        return self.snapshot()

    @staticmethod
    def export_standalone_html(out_path: Optional[str] = None) -> str:
        """Write the page as a single file. Opened without an engine it runs as a labelled simulation."""
        p = Path(resolve_output(out_path, "rcwe_mission_control.html"))
        p.write_text(HTML_TEMPLATE, encoding="utf-8")
        return str(p.resolve())


def serve_mission_control(host: str = "127.0.0.1", port: int = 8766, open_browser: bool = True) -> None:
    """Start the local server and open Mission Control (Studio is served from the same address)."""
    from rcwe.toolkit.studio import serve

    serve(host=host, port=port, open_browser=open_browser, start_page="/mission-control")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RCWE Mission Control")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--export-only", action="store_true", help="write the standalone page and exit")
    args = parser.parse_args()
    if args.export_only:
        print(f"Exported standalone Mission Control page -> {MissionControlEngine.export_standalone_html()}")
    else:
        serve_mission_control(host=args.host, port=args.port)
