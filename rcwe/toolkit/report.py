"""
Self-contained HTML report for a `DetectionResult`.

One file, no network access needed to open it: styles, data and chart code are inlined.
Every number on the page comes from the result object (and, through it, from the model).

    result = model.detect("today.csv")
    result.to_html("report.html")

Column names, file names and notes are treated as untrusted text: they travel to the page
inside an escaped JSON block and are inserted with `textContent`, never as HTML.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from rcwe.toolkit.results import DetectionResult
from rcwe.toolkit.scoring import find_events

__all__ = ["build_report_data", "render_report", "write_report"]


def _num(value: Any, digits: int = 5) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(f"{value:.{digits}g}")


def _series(values: np.ndarray, digits: int = 5) -> List[Optional[float]]:
    return [_num(v, digits) for v in values]


def _keep_rows(result: DetectionResult, max_points: int) -> np.ndarray:
    """Rows to embed: everything for short logs; for long ones, full detail around events and
    one or two representative rows (always including the highest score) per bucket elsewhere."""
    T = result.n_rows
    if T <= int(max_points * 1.5):
        return np.arange(T)
    keep = {0, T - 1}
    bucket = int(math.ceil(T / max_points))
    for start in range(0, T, bucket):
        stop = min(start + bucket, T)
        keep.add(start)
        keep.add(start + int(np.argmax(result.scores[start:stop])))
    budget = 2 * max_points
    for event in sorted(result.events, key=lambda e: -e.peak_score):
        pad = max(20, event.duration)
        lo, hi = max(event.start - pad, 0), min(event.end + pad, T - 1)
        window = range(lo, hi + 1)
        stride = max(1, len(window) // 400)
        chosen = set(window[::stride]) | {event.start, event.end, event.peak}
        if len(chosen) > budget:
            continue
        budget -= len(chosen)
        keep |= chosen
    return np.array(sorted(keep), dtype=int)


def build_report_data(
    result: DetectionResult,
    title: Optional[str] = None,
    max_points: int = 1800,
    max_panels: int = 12,
) -> Dict[str, Any]:
    """Assemble the JSON-serialisable payload the report page renders."""
    from rcwe import __version__

    rows = _keep_rows(result, max_points)
    ranking = result.sensor_ranking()
    order = [result.sensors.index(r["name"]) for r in ranking]
    top_sensor = np.argmax(result.sensor_scores, axis=1) if result.n_rows else np.zeros(0, dtype=int)

    panels = []
    for i in order[:max_panels]:
        flagged = find_events(result.sensor_scores[:, i], result.threshold)
        panels.append({
            "name": result.sensors[i],
            "actual": _series(result.values[rows, i]),
            "expected": _series(result.expected[rows, i]),
            "z": _series(result.residual_z[rows, i], 3),
            "band": _num(result.band[i]),
            "flagged": [[int(a), int(b)] for a, b in flagged],
        })

    forecast = None
    if result.forecast is not None:
        fc = result.forecast
        fc_panels = []
        for name in [result.sensors[i] for i in order[:max_panels]]:
            if name not in fc.sensors:
                continue
            j = fc.sensors.index(name)
            fc_panels.append({
                "name": name,
                "history": [] if fc.history is None else _series(fc.history[:, j]),
                "values": _series(fc.values[:, j]),
                "lower": _series(fc.lower[:, j]),
                "upper": _series(fc.upper[:, j]),
                "skill": _num(fc.sensor_skill.get(name), 3),
                "weight": _num(fc.model_weight.get(name), 3),
            })
        forecast = {
            "steps": fc.steps, "origin": fc.origin, "timestamps": list(fc.timestamps),
            "history_timestamps": fc.history_timestamps or [], "panels": fc_panels, "notes": list(fc.notes),
        }

    model = dict(result.model_info)
    if isinstance(model.get("source"), str) and not model["source"].startswith("<"):
        model["source"] = Path(model["source"]).name
    return {
        "title": title or "RCWE anomaly report",
        # A report is something people pass around: show the file's name, not where it sat on disk.
        "source": Path(result.source).name if result.source and not result.source.startswith("<") else (result.source or ""),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "version": __version__,
        "mode": result.mode,
        "rows": result.n_rows,
        "sensors": list(result.sensors),
        "start": result.timestamps[0] if result.timestamps else "",
        "end": result.timestamps[-1] if result.timestamps else "",
        "threshold": _num(result.threshold),
        "peak": _num(float(result.scores.max()) if result.n_rows else 0.0),
        "anomalous_fraction": _num(result.anomalous_fraction),
        "events": [e.to_dict() for e in result.events],
        "metrics": None if result.metrics is None else {k: (_num(v) if isinstance(v, float) else v) for k, v in result.metrics.items()},
        "notes": list(result.notes),
        "model": model,
        "index": [int(r) for r in rows],
        "time": [result.timestamps[r] for r in rows],
        "score": _series(result.scores[rows], 4),
        "top": [int(top_sensor[r]) for r in rows],
        "labels": None if result.labels is None else [int(result.labels[r] > 0) for r in rows],
        "panels": panels,
        "hidden_sensors": max(len(result.sensors) - len(panels), 0),
        "ranking": [
            {"name": r["name"], "steps": int(r["anomalous_steps"]), "peak": _num(r["peak_score"], 4)} for r in ranking
        ],
        "forecast": forecast,
    }


def render_report(result: DetectionResult, title: Optional[str] = None) -> str:
    """Return the report as an HTML string."""
    data = build_report_data(result, title=title)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    # The payload sits inside a <script> block: make sure no column name can close it early.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page_title = (title or "RCWE anomaly report").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _TEMPLATE.replace("__TITLE__", page_title).replace("__DATA__", payload)


def write_report(result: DetectionResult, path: Union[str, Path], title: Optional[str] = None) -> str:
    """Write the report to `path` and return the absolute path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(result, title=title), encoding="utf-8")
    return str(out.resolve())


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --wash: rgba(11,11,11,0.04);
    --series: #2a78d6; --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
    --band: rgba(137,135,129,0.18); --event: rgba(208,59,59,0.13); --truth: rgba(82,81,78,0.55);
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.05);
      --series: #3987e5; --band: rgba(137,135,129,0.22); --event: rgba(208,59,59,0.22); --truth: rgba(195,194,183,0.6);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.05);
    --series: #3987e5; --band: rgba(137,135,129,0.22); --event: rgba(208,59,59,0.22); --truth: rgba(195,194,183,0.6);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 56px; }
  header.top { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 20px; }
  h1 { font-size: 22px; line-height: 1.25; margin: 0 0 4px; font-weight: 650; }
  h2 { font-size: 16px; margin: 0 0 2px; font-weight: 650; }
  .sub { color: var(--ink2); font-size: 13px; overflow-wrap: anywhere; }
  .hint { color: var(--muted); font-size: 12px; }
  button, select { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer; }
  button:hover { background: var(--wash); }
  button[aria-pressed="true"] { border-color: var(--series); box-shadow: inset 0 0 0 1px var(--series); }
  button:focus-visible, tr:focus-visible, select:focus-visible { outline: 2px solid var(--series); outline-offset: 1px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; margin-bottom: 16px; }
  .verdict { display: flex; flex-wrap: wrap; gap: 28px; align-items: center; }
  .hero { font-size: 56px; line-height: 1; font-weight: 650; letter-spacing: -0.02em; }
  .verdict .lead { font-size: 17px; font-weight: 600; }
  .status { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; white-space: nowrap; }
  .status .mark { width: 16px; height: 16px; flex: none; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }
  .tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .tile .label { color: var(--ink2); font-size: 12px; }
  .tile .value { font-size: 24px; font-weight: 650; line-height: 1.25; }
  .tile .note { color: var(--muted); font-size: 12px; }
  .viewbar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 4px 0 14px; }
  .viewbar .lbl { color: var(--ink2); font-size: 13px; margin-right: 4px; }
  .chart { position: relative; width: 100%; }
  .chart svg { display: block; width: 100%; overflow: visible; }
  .chart text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--muted); font-variant-numeric: tabular-nums; }
  .chart text.strong { fill: var(--ink2); }
  .chart text.halo { paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round; }
  .legend { display: flex; flex-wrap: wrap; gap: 16px; color: var(--ink2); font-size: 12px; margin: 6px 0 10px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .key-line { width: 18px; height: 0; border-top: 2px solid var(--series); }
  .key-line.expected { border-top: 2px solid var(--muted); }
  .key-box { width: 14px; height: 10px; border-radius: 2px; background: var(--band); }
  .key-box.event { background: var(--event); }
  .key-box.truth { background: var(--truth); height: 4px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px 24px; }
  .panel h3 { font-size: 13px; margin: 0; font-weight: 600; overflow-wrap: anywhere; }
  .panel .hint { margin-bottom: 2px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  #events { min-width: 760px; }
  th, td { text-align: left; padding: 7px 10px 7px 0; border-bottom: 1px solid var(--grid); vertical-align: top; }
  th { color: var(--ink2); font-weight: 600; font-size: 12px; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  tbody tr.click { cursor: pointer; }
  tbody tr.click:hover, tbody tr.sel { background: var(--wash); }
  .scroll { overflow-x: auto; }
  .bars { display: grid; grid-template-columns: minmax(110px, max-content) 1fr max-content; gap: 8px 12px; align-items: center; font-size: 13px; }
  .bars .name { overflow-wrap: anywhere; }
  .bars .track { height: 16px; display: flex; align-items: center; }
  .bars .bar { height: 14px; min-width: 2px; background: var(--series); border-radius: 0 4px 4px 0; }
  .bars .bar:hover { filter: brightness(1.12); }
  .bars .val { color: var(--ink2); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .meter { display: inline-block; vertical-align: middle; width: 90px; height: 8px; border-radius: 4px; background: color-mix(in srgb, var(--series) 18%, transparent); margin-right: 8px; overflow: hidden; }
  .meter i { display: block; height: 100%; background: var(--series); border-radius: 4px; }
  ul.notes { margin: 6px 0 0; padding-left: 18px; color: var(--ink2); }
  ul.notes li { margin: 3px 0; overflow-wrap: anywhere; }
  details summary { cursor: pointer; color: var(--ink2); font-size: 13px; margin-top: 10px; }
  #tip { position: fixed; z-index: 20; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 6px 24px rgba(0,0,0,0.18); padding: 8px 10px; font-size: 12px; max-width: 320px; display: none; }
  #tip .t { color: var(--ink2); margin-bottom: 4px; }
  #tip .row { display: flex; gap: 8px; align-items: center; justify-content: space-between; }
  #tip .row b { font-weight: 650; font-variant-numeric: tabular-nums; }
  #tip .row span { color: var(--ink2); display: inline-flex; gap: 6px; align-items: center; }
  footer { color: var(--muted); font-size: 12px; margin-top: 22px; }
  @media print { button, .viewbar { display: none; } .card { break-inside: avoid; } }
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <h1 id="title"></h1>
      <div class="sub" id="subtitle"></div>
    </div>
    <button id="theme" type="button" aria-label="Switch between light and dark theme">Theme</button>
  </header>

  <section class="card verdict" aria-live="polite">
    <div class="hero" id="hero"></div>
    <div>
      <div class="lead" id="lead"></div>
      <div class="sub" id="leadsub"></div>
    </div>
  </section>

  <section class="tiles" id="tiles"></section>

  <section class="card">
    <h2>Anomaly score over time</h2>
    <div class="sub">Worst sensor deviation at each step, in calibrated sigmas. Shaded spans are the detected events.</div>
    <div class="viewbar" id="viewbar" role="group" aria-label="Time range shown in every chart"></div>
    <div class="legend" id="scorelegend"></div>
    <div class="chart" id="scorechart"></div>
    <details id="scoretable"><summary>Show flagged rows as a table</summary><div class="scroll"></div></details>
  </section>

  <section class="card" id="eventcard">
    <h2>Detected events</h2>
    <div class="sub">Select a row to zoom every chart to that event.</div>
    <div class="scroll"><table id="events"><thead><tr>
      <th class="num">#</th><th>Severity</th><th>When</th><th class="num">Steps</th><th>Type</th><th>Main sensor</th><th>What happened</th>
    </tr></thead><tbody></tbody></table></div>
  </section>

  <section class="card" id="rankcard">
    <h2>Which sensors were involved</h2>
    <div class="sub">Number of steps each sensor spent above the alarm threshold.</div>
    <div class="bars" id="ranking" style="margin-top:12px"></div>
  </section>

  <section class="card">
    <h2>Readings against what the model expected</h2>
    <div class="sub" id="panelsub"></div>
    <div class="legend">
      <span><i class="key-line"></i>Actual reading</span>
      <span><i class="key-line expected"></i>Expected (one step ahead)</span>
      <span><i class="key-box"></i>Normal band</span>
      <span><i class="key-box event"></i>Sensor above threshold</span>
    </div>
    <div class="grid" id="panels"></div>
  </section>

  <section class="card" id="forecastcard" hidden>
    <h2>Forecast</h2>
    <div class="sub" id="forecastsub"></div>
    <div class="legend">
      <span><i class="key-line expected"></i>Recent history</span>
      <span><i class="key-line"></i>Forecast</span>
      <span><i class="key-box" style="background:color-mix(in srgb, var(--series) 16%, transparent)"></i>80% interval</span>
    </div>
    <div class="grid" id="forecastpanels"></div>
    <ul class="notes" id="forecastnotes"></ul>
  </section>

  <section class="card">
    <h2>How much to trust this</h2>
    <div class="sub" id="trustsub"></div>
    <div class="scroll" style="margin-top:10px"><table id="skill"><thead><tr>
      <th>Sensor</th><th>One-step skill vs naive baseline</th><th class="num">Steps above threshold</th><th class="num">Peak score</th>
    </tr></thead><tbody></tbody></table></div>
    <ul class="notes" id="notes"></ul>
  </section>

  <footer id="footer"></footer>
</div>
<div id="tip" role="status"></div>
<script type="application/json" id="rcwe-data">__DATA__</script>
<script>
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("rcwe-data").textContent);
  var NS = "http://www.w3.org/2000/svg";
  var view = [0, Math.max(D.rows - 1, 1)];
  var charts = [];
  var tip = document.getElementById("tip");

  function h(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function s(tag, attrs) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function fmt(v) {
    if (v === null || v === undefined || !isFinite(v)) return "n/a";
    var a = Math.abs(v);
    if (a >= 10000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (a >= 100) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return "0";
    return v.toPrecision(3);
  }
  function fmtTick(v, step) {
    if (Math.abs(v) >= 10000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    var d = step >= 1 ? 0 : Math.min(6, Math.ceil(-Math.log10(step)));
    return v.toFixed(d);
  }
  function tickStep(list) { return list.length > 1 ? Math.abs(list[1] - list[0]) : 1; }
  function pct(v) { return (v === null || v === undefined || !isFinite(v)) ? "n/a" : (v >= 0 ? "+" : "") + Math.round(v * 100) + "%"; }
  function ticks(lo, hi, n) {
    if (!(hi > lo)) { hi = lo + 1; }
    var raw = (hi - lo) / Math.max(n, 1), mag = Math.pow(10, Math.floor(Math.log10(raw))), norm = raw / mag;
    var step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag, out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  function shortTime(label, sameDay) {
    var m = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}:\d{2})(:\d{2})?/.exec(label);
    if (!m) return label.length > 16 ? label.slice(0, 16) : label;
    return sameDay ? m[4] + (m[5] && m[5] !== ":00" ? m[5] : "") : m[2] + "-" + m[3] + " " + m[4];
  }
  function lowerBound(arr, x) { var lo = 0, hi = arr.length; while (lo < hi) { var mid = (lo + hi) >> 1; if (arr[mid] < x) lo = mid + 1; else hi = mid; } return lo; }
  function nearest(arr, x) {
    var i = lowerBound(arr, x);
    if (i <= 0) return 0;
    if (i >= arr.length) return arr.length - 1;
    return (x - arr[i - 1] <= arr[i] - x) ? i - 1 : i;
  }
  function showTip(evt, title, rows) {
    tip.textContent = "";
    tip.appendChild(h("div", "t", title));
    rows.forEach(function (r) {
      var row = h("div", "row"), left = h("span");
      if (r.key) { var k = h("i", "key-line" + (r.key === "expected" ? " expected" : "")); left.appendChild(k); }
      left.appendChild(document.createTextNode(r.label));
      row.appendChild(left); row.appendChild(h("b", null, r.value));
      tip.appendChild(row);
    });
    tip.style.display = "block";
    var w = tip.offsetWidth, hh = tip.offsetHeight, x = evt.clientX + 14, y = evt.clientY + 14;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - 14;
    if (y + hh > window.innerHeight - 8) y = evt.clientY - hh - 14;
    tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { tip.style.display = "none"; }

  /* A time chart over the shared row axis. opts: xs (row numbers), labels, lines [{v, cls, w}],
     band {v, half}, spans [[a, b]], spanLabels, truth [0/1], threshold, zeroBased, tooltip(i) */
  function timeChart(host, opts) {
    var H = opts.height || 190, M = { l: 46, r: 14, t: 14, b: 26 };
    var svgEl = s("svg", { role: "img", "aria-label": opts.label || "chart" });
    host.appendChild(svgEl);
    function draw() {
      var W = Math.max(host.clientWidth, 260), iw = W - M.l - M.r, ih = H - M.t - M.b;
      svgEl.setAttribute("viewBox", "0 0 " + W + " " + H); svgEl.setAttribute("height", H);
      while (svgEl.firstChild) svgEl.removeChild(svgEl.firstChild);
      var xs = opts.xs, i0 = lowerBound(xs, view[0]), i1 = Math.min(lowerBound(xs, view[1] + 1), xs.length) - 1;
      if (i1 <= i0) { i0 = Math.max(0, Math.min(i0, xs.length - 2)); i1 = Math.min(xs.length - 1, i0 + 1); }
      var x0 = view[0], x1 = Math.max(view[1], view[0] + 1);
      function X(row) { return M.l + (row - x0) / (x1 - x0) * iw; }
      var lo = Infinity, hi = -Infinity;
      opts.lines.forEach(function (ln) { for (var i = i0; i <= i1; i++) { var v = ln.v[i]; if (v === null) continue; if (v < lo) lo = v; if (v > hi) hi = v; } });
      if (opts.band) for (var i = i0; i <= i1; i++) { var c = opts.band.v[i]; if (c === null) continue; lo = Math.min(lo, c - opts.band.half); hi = Math.max(hi, c + opts.band.half); }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      var clampTop = null;
      if (opts.zeroBased) {
        lo = 0;
        var cap = Math.max(opts.threshold * 4, 1);
        if (hi > cap * 1.6) { clampTop = cap * 1.6; hi = clampTop; }
        hi = Math.max(hi, opts.threshold * 1.35);
      } else { var pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1; lo -= pad; hi += pad; }
      function Y(v) { return M.t + (1 - (Math.min(v, hi) - lo) / (hi - lo)) * ih; }

      var yt = ticks(lo, hi, 4), ystep = tickStep(yt);
      yt.forEach(function (v) {
        svgEl.appendChild(s("line", { x1: M.l, x2: M.l + iw, y1: Y(v), y2: Y(v), stroke: "var(--grid)", "stroke-width": 1 }));
        var t = s("text", { x: M.l - 8, y: Y(v) + 3.5, "text-anchor": "end" }); t.textContent = fmtTick(v, ystep); svgEl.appendChild(t);
      });
      svgEl.appendChild(s("line", { x1: M.l, x2: M.l + iw, y1: M.t + ih, y2: M.t + ih, stroke: "var(--axis)", "stroke-width": 1 }));
      var sameDay = opts.labels[i0].slice(0, 10) === opts.labels[i1].slice(0, 10);
      var nt = Math.max(2, Math.min(6, Math.floor(iw / 110)));
      for (var k = 0; k <= nt; k++) {
        var idx = nearest(xs, x0 + (x1 - x0) * k / nt);
        var tx = s("text", { x: X(xs[idx]), y: H - 8, "text-anchor": k === 0 ? "start" : k === nt ? "end" : "middle" });
        tx.textContent = shortTime(opts.labels[idx], sameDay); svgEl.appendChild(tx);
      }

      var lastLabelX = -Infinity;
      (opts.spans || []).forEach(function (sp, n) {
        if (sp[1] < x0 || sp[0] > x1) return;
        var a = Math.max(X(sp[0] - 0.5), M.l), b = Math.min(X(sp[1] + 0.5), M.l + iw);
        svgEl.appendChild(s("rect", { x: a, y: M.t, width: Math.max(b - a, 2), height: ih, fill: "var(--event)" }));
        if (opts.spanLabels && (a + b) / 2 - lastLabelX > 22) { lastLabelX = (a + b) / 2; var lb = s("text", { x: (a + b) / 2, y: M.t - 3, "text-anchor": "middle", class: "strong" }); lb.textContent = "#" + opts.spanLabels[n]; svgEl.appendChild(lb); }
      });
      if (opts.truth) {
        var run = -1;
        for (var i = i0; i <= i1 + 1; i++) {
          var on = i <= i1 && opts.truth[i] === 1;
          if (on && run < 0) run = i;
          if (!on && run >= 0) { var ta = X(xs[run] - 0.5), tb = X(xs[i - 1] + 0.5); svgEl.appendChild(s("rect", { x: ta, y: M.t + ih + 2, width: Math.max(tb - ta, 2), height: 4, rx: 2, fill: "var(--truth)" })); run = -1; }
        }
      }
      if (opts.band) {
        var up = "", dn = "";
        for (var i = i0; i <= i1; i++) { var c = opts.band.v[i]; if (c === null) continue; up += (up ? "L" : "M") + X(xs[i]).toFixed(1) + "," + Y(c + opts.band.half).toFixed(1); dn = "L" + X(xs[i]).toFixed(1) + "," + Y(c - opts.band.half).toFixed(1) + dn; }
        if (up) svgEl.appendChild(s("path", { d: up + dn + "Z", fill: "var(--band)" }));
      }
      opts.lines.forEach(function (ln) {
        var d = "", pen = false;
        for (var i = i0; i <= i1; i++) { var v = ln.v[i]; if (v === null) { pen = false; continue; } d += (pen ? "L" : "M") + X(xs[i]).toFixed(1) + "," + Y(v).toFixed(1); pen = true; }
        svgEl.appendChild(s("path", { d: d, fill: "none", stroke: ln.cls === "expected" ? "var(--muted)" : "var(--series)", "stroke-width": ln.w || 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      });
      if (opts.threshold !== undefined) {
        svgEl.appendChild(s("line", { x1: M.l, x2: M.l + iw, y1: Y(opts.threshold), y2: Y(opts.threshold), stroke: "var(--ink2)", "stroke-width": 1, "stroke-dasharray": "5 4" }));
        var tl = s("text", { x: M.l + iw, y: Y(opts.threshold) - 5, "text-anchor": "end", class: "strong halo" }); tl.textContent = "alarm threshold " + fmt(opts.threshold); svgEl.appendChild(tl);
      }
      if (clampTop !== null) { var ct = s("text", { x: M.l + 4, y: M.t + 10, class: "strong" }); ct.textContent = "scale capped; peak " + fmt(D.peak); svgEl.appendChild(ct); }

      var cross = s("line", { y1: M.t, y2: M.t + ih, stroke: "var(--ink2)", "stroke-width": 1, visibility: "hidden" });
      svgEl.appendChild(cross);
      var dots = opts.lines.map(function (ln) { var c = s("circle", { r: 4, fill: ln.cls === "expected" ? "var(--muted)" : "var(--series)", stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }); svgEl.appendChild(c); return c; });
      var hit = s("rect", { x: M.l, y: M.t, width: iw, height: ih, fill: "transparent", tabindex: 0, "aria-label": (opts.label || "chart") + ". Use arrow keys to read values." });
      svgEl.appendChild(hit);
      var cur = -1;
      function focusAt(i, evt) {
        cur = Math.max(i0, Math.min(i1, i));
        var cx = X(xs[cur]);
        cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
        opts.lines.forEach(function (ln, n) { var v = ln.v[cur]; if (v === null) { dots[n].setAttribute("visibility", "hidden"); return; } dots[n].setAttribute("cx", cx); dots[n].setAttribute("cy", Y(v)); dots[n].setAttribute("visibility", "visible"); });
        if (!evt) { var box = svgEl.getBoundingClientRect(); evt = { clientX: box.left + cx * box.width / W, clientY: box.top + M.t + 8 }; }
        var info = opts.tooltip(cur); showTip(evt, info.title, info.rows);
      }
      function leave() { cross.setAttribute("visibility", "hidden"); dots.forEach(function (c) { c.setAttribute("visibility", "hidden"); }); hideTip(); }
      hit.addEventListener("pointermove", function (evt) { var box = svgEl.getBoundingClientRect(); var px = (evt.clientX - box.left) * W / box.width; focusAt(nearest(xs, x0 + (px - M.l) / iw * (x1 - x0)), evt); });
      hit.addEventListener("pointerleave", leave);
      hit.addEventListener("blur", leave);
      hit.addEventListener("focus", function () { focusAt(cur < 0 ? i0 : cur); });
      hit.addEventListener("keydown", function (evt) { if (evt.key === "ArrowRight") { focusAt(cur + 1); evt.preventDefault(); } else if (evt.key === "ArrowLeft") { focusAt(cur - 1); evt.preventDefault(); } });
    }
    charts.push(draw);
    draw();
  }

  function statusMark(level) {
    var color = { good: "var(--good)", low: "var(--warning)", medium: "var(--serious)", high: "var(--critical)", critical: "var(--critical)" }[level];
    var g = s("svg", { viewBox: "0 0 16 16", class: "mark", "aria-hidden": "true" });
    if (level === "good") { g.appendChild(s("circle", { cx: 8, cy: 8, r: 7, fill: color })); g.appendChild(s("path", { d: "M4.5 8.3l2.3 2.3 4.7-5", fill: "none", stroke: "#fff", "stroke-width": 1.8, "stroke-linecap": "round", "stroke-linejoin": "round" })); }
    else if (level === "low") { g.appendChild(s("path", { d: "M8 2l6.5 11.5h-13z", fill: color })); }
    else if (level === "medium") { g.appendChild(s("rect", { x: 2.5, y: 2.5, width: 11, height: 11, rx: 2, fill: color })); }
    else if (level === "high") { g.appendChild(s("circle", { cx: 8, cy: 8, r: 6.5, fill: color })); }
    else { g.appendChild(s("path", { d: "M8 1l7 7-7 7-7-7z", fill: color })); g.appendChild(s("path", { d: "M8 4.5v4.2M8 11v.6", stroke: "#fff", "stroke-width": 1.8, "stroke-linecap": "round" })); }
    return g;
  }
  function status(level, text) { var w = h("span", "status"); w.appendChild(statusMark(level)); w.appendChild(document.createTextNode(text)); return w; }
  function cap(t) { return t.charAt(0).toUpperCase() + t.slice(1); }

  /* ---------- header, verdict, tiles ---------- */
  document.getElementById("title").textContent = D.title;
  document.getElementById("subtitle").textContent = D.source + "  ·  " + D.start + " to " + D.end + "  ·  " + D.rows.toLocaleString() + " rows × " + D.sensors.length + " sensors";
  var n = D.events.length, worst = "good";
  var rank = { low: 1, medium: 2, high: 3, critical: 4 };
  D.events.forEach(function (e) { if (worst === "good" || rank[e.severity] > rank[worst]) worst = e.severity; });
  document.getElementById("hero").textContent = n;
  var lead = document.getElementById("lead");
  lead.appendChild(status(worst, n === 0 ? "No anomalies detected" : n === 1 ? "anomaly event detected" : "anomaly events detected"));
  document.getElementById("leadsub").textContent = n === 0
    ? "The highest score was " + fmt(D.peak) + " sigma, below the alarm threshold of " + fmt(D.threshold) + "."
    : (D.anomalous_fraction * 100).toFixed(1) + "% of the log is inside an event. Most severe: " + cap(worst) + ". " + (D.mode === "adaptive" ? "Adaptive mode: events mark the onset of changes." : "Reference mode: readings are compared with the behaviour learned in training.");

  var tiles = document.getElementById("tiles");
  function tile(label, value, note) { var t = h("div", "tile"); t.appendChild(h("div", "label", label)); t.appendChild(h("div", "value", value)); if (note) t.appendChild(h("div", "note", note)); tiles.appendChild(t); }
  tile("Rows scanned", D.rows.toLocaleString(), D.sensors.length + " sensors");
  tile("Peak score", fmt(D.peak), "alarm threshold " + fmt(D.threshold));
  tile("Time inside events", (D.anomalous_fraction * 100).toFixed(1) + "%", n + (n === 1 ? " event" : " events"));
  var skill = D.model.skill_vs_persistence;
  if (skill !== undefined && skill !== null) tile("Model skill", pct(skill), "one step ahead, vs naive baseline");
  if (D.metrics && D.metrics.labeled_incidents) {
    tile("Labelled incidents caught", D.metrics.incidents_detected + " of " + D.metrics.labeled_incidents, D.metrics.false_alarm_events + " false alarm" + (D.metrics.false_alarm_events === 1 ? "" : "s"));
    if (D.metrics.auroc !== null && D.metrics.auroc !== undefined) tile("AUROC vs labels", D.metrics.auroc.toFixed(3), "F1 " + D.metrics.point_f1.toFixed(2) + " point-wise");
  }

  /* ---------- shared view control ---------- */
  var viewbar = document.getElementById("viewbar");
  function setView(a, b, key) {
    view = [Math.max(0, a), Math.min(D.rows - 1, b)];
    Array.prototype.forEach.call(viewbar.querySelectorAll("button"), function (btn) { btn.setAttribute("aria-pressed", btn.dataset.key === key ? "true" : "false"); });
    Array.prototype.forEach.call(document.querySelectorAll("#events tbody tr"), function (tr) { tr.classList.toggle("sel", tr.dataset.key === key); });
    charts.forEach(function (draw) { draw(); });
  }
  function zoomTo(e) { var pad = Math.max(12, Math.round((e.end - e.start + 1) * 1.5)); setView(e.start - pad, e.end + pad, "e" + e.index); }
  viewbar.appendChild(h("span", "lbl", "Show"));
  var all = h("button", null, "Whole log"); all.type = "button"; all.dataset.key = "all"; all.setAttribute("aria-pressed", "true");
  all.addEventListener("click", function () { setView(0, D.rows - 1, "all"); }); viewbar.appendChild(all);
  D.events.slice(0, 14).forEach(function (e) { var b = h("button", null, "Event " + e.index); b.type = "button"; b.dataset.key = "e" + e.index; b.setAttribute("aria-pressed", "false"); b.addEventListener("click", function () { zoomTo(e); }); viewbar.appendChild(b); });
  if (D.events.length > 14) viewbar.appendChild(h("span", "hint", "…more in the table below"));

  /* ---------- score timeline ---------- */
  var legend = document.getElementById("scorelegend");
  function legendItem(cls, text) { var sp = h("span"); sp.appendChild(h("i", cls)); sp.appendChild(document.createTextNode(text)); legend.appendChild(sp); }
  legendItem("key-box event", "Detected event");
  if (D.labels) legendItem("key-box truth", "Labelled as anomalous in the file");
  timeChart(document.getElementById("scorechart"), {
    xs: D.index, labels: D.time, height: 230, label: "Anomaly score over time",
    lines: [{ v: D.score, cls: "score", w: 2 }], threshold: D.threshold, zeroBased: true,
    spans: D.events.map(function (e) { return [e.start, e.end]; }), spanLabels: D.events.map(function (e) { return e.index; }), truth: D.labels,
    tooltip: function (i) {
      var rows = [{ label: "Anomaly score", value: fmt(D.score[i]), key: "actual" }];
      if (D.score[i] > D.threshold) rows.push({ label: "Worst sensor", value: D.sensors[D.top[i]] });
      if (D.labels) rows.push({ label: "Labelled anomalous", value: D.labels[i] ? "yes" : "no" });
      return { title: D.time[i], rows: rows };
    }
  });
  (function () {
    var holder = document.querySelector("#scoretable .scroll"), table = h("table"), head = h("tr"), body = h("tbody"), shown = 0;
    ["Time", "Anomaly score", "Worst sensor"].forEach(function (c, k) { var th = h("th", k === 1 ? "num" : null, c); head.appendChild(th); });
    var thead = h("thead"); thead.appendChild(head); table.appendChild(thead); table.appendChild(body);
    for (var i = 0; i < D.index.length && shown < 400; i++) { if (D.score[i] > D.threshold) { var tr = h("tr"); tr.appendChild(h("td", null, D.time[i])); tr.appendChild(h("td", "num", fmt(D.score[i]))); tr.appendChild(h("td", null, D.sensors[D.top[i]])); body.appendChild(tr); shown++; } }
    if (!shown) { var tr0 = h("tr"); var td0 = h("td", null, "No row is above the alarm threshold."); td0.colSpan = 3; tr0.appendChild(td0); body.appendChild(tr0); }
    holder.appendChild(table);
    if (shown >= 400) holder.appendChild(h("div", "hint", "First 400 flagged rows shown. Export the full per-row table with result.to_csv() or `rcwe detect --csv`."));
  })();

  /* ---------- events table ---------- */
  var tbody = document.querySelector("#events tbody");
  if (!D.events.length) document.getElementById("eventcard").hidden = true;
  D.events.forEach(function (e) {
    var tr = h("tr", "click"); tr.tabIndex = 0; tr.dataset.key = "e" + e.index;
    tr.appendChild(h("td", "num", e.index));
    var sev = h("td"); sev.appendChild(status(e.severity, cap(e.severity))); tr.appendChild(sev);
    tr.appendChild(h("td", null, e.start_time === e.end_time ? e.start_time : e.start_time + " → " + e.end_time));
    tr.appendChild(h("td", "num", e.duration_steps));
    tr.appendChild(h("td", null, cap(e.kind)));
    tr.appendChild(h("td", null, e.sensors.length ? e.sensors[0].name : "–"));
    tr.appendChild(h("td", null, e.what || e.description));
    tr.addEventListener("click", function () { zoomTo(e); document.getElementById("scorechart").scrollIntoView({ behavior: "smooth", block: "center" }); });
    tr.addEventListener("keydown", function (evt) { if (evt.key === "Enter" || evt.key === " ") { evt.preventDefault(); zoomTo(e); } });
    tbody.appendChild(tr);
  });

  /* ---------- sensor ranking ---------- */
  var ranking = document.getElementById("ranking"), involved = D.ranking.filter(function (r) { return r.steps > 0; });
  if (!involved.length) document.getElementById("rankcard").hidden = true;
  var maxSteps = Math.max.apply(null, involved.map(function (r) { return r.steps; }).concat([1]));
  involved.slice(0, 15).forEach(function (r) {
    ranking.appendChild(h("div", "name", r.name));
    var track = h("div", "track"), bar = h("div", "bar"); bar.style.width = (r.steps / maxSteps * 100) + "%"; bar.tabIndex = 0;
    function tipOn(evt) { var box = bar.getBoundingClientRect(); showTip(evt.clientX ? evt : { clientX: box.right, clientY: box.top }, r.name, [{ label: "Steps above threshold", value: r.steps.toLocaleString() }, { label: "Peak score", value: fmt(r.peak) }]); }
    bar.addEventListener("pointermove", tipOn); bar.addEventListener("pointerleave", hideTip); bar.addEventListener("focus", tipOn); bar.addEventListener("blur", hideTip);
    track.appendChild(bar); ranking.appendChild(track);
    ranking.appendChild(h("div", "val", r.steps.toLocaleString() + (r.steps === 1 ? " step" : " steps")));
  });

  /* ---------- sensor panels ---------- */
  document.getElementById("panelsub").textContent = "Sensors are ordered by involvement. The band is the range within which a reading counts as normal."
    + (D.hidden_sensors ? " Showing the " + D.panels.length + " most involved of " + D.sensors.length + " sensors." : "");
  var panels = document.getElementById("panels");
  D.panels.forEach(function (p) {
    var box = h("div", "panel"); box.appendChild(h("h3", null, p.name));
    var steps = (D.ranking.filter(function (r) { return r.name === p.name; })[0] || { steps: 0 }).steps;
    box.appendChild(h("div", "hint", steps ? steps.toLocaleString() + " step" + (steps === 1 ? "" : "s") + " above threshold" : "within the normal band throughout"));
    var host = h("div", "chart"); box.appendChild(host); panels.appendChild(box);
    timeChart(host, {
      xs: D.index, labels: D.time, height: 170, label: p.name + ": actual and expected readings",
      lines: [{ v: p.expected, cls: "expected", w: 1.5 }, { v: p.actual, cls: "actual", w: 2 }],
      band: { v: p.expected, half: p.band }, spans: p.flagged,
      tooltip: function (i) { return { title: D.time[i], rows: [{ label: "Actual", value: fmt(p.actual[i]), key: "actual" }, { label: "Expected", value: fmt(p.expected[i]), key: "expected" }, { label: "Surprise", value: (p.z[i] >= 0 ? "+" : "") + fmt(p.z[i]) + " sigma" }] }; }
    });
  });

  /* ---------- forecast ---------- */
  if (D.forecast && D.forecast.panels.length) {
    var F = D.forecast; document.getElementById("forecastcard").hidden = false;
    document.getElementById("forecastsub").textContent = F.steps + " steps ahead from " + F.origin + ". Intervals come from backtests on held-out data; where a sensor shows no forecast skill the forecast simply holds its recent behaviour.";
    var fhost = document.getElementById("forecastpanels");
    F.panels.forEach(function (p) {
      var nh = p.history.length, labels = F.history_timestamps.slice(0, nh).concat(F.timestamps), xs = [], hist = [], fc = [];
      for (var i = 0; i < nh + F.steps; i++) { xs.push(i); hist.push(i < nh ? p.history[i] : null); fc.push(i >= nh ? p.values[i - nh] : (i === nh - 1 ? p.history[i] : null)); }
      while (labels.length < xs.length) labels.unshift("");
      var box = h("div", "panel"); box.appendChild(h("h3", null, p.name));
      box.appendChild(h("div", "hint", p.weight !== null && p.weight < 0.05 ? "no skill over the naive baseline: holding recent behaviour" : "backtested skill " + pct(p.skill) + " vs naive baseline"));
      var host = h("div", "chart"); box.appendChild(host); fhost.appendChild(box);
      /* forecast charts keep their own fixed window: history + horizon */
      (function () {
        var H = 170, M = { l: 46, r: 14, t: 14, b: 26 }, svgEl = s("svg", { role: "img", "aria-label": p.name + " forecast" }); host.appendChild(svgEl);
        function render() {
          var W = Math.max(host.clientWidth, 260), iw = W - M.l - M.r, ih = H - M.t - M.b, N = xs.length;
          svgEl.setAttribute("viewBox", "0 0 " + W + " " + H); svgEl.setAttribute("height", H);
          while (svgEl.firstChild) svgEl.removeChild(svgEl.firstChild);
          var lo = Infinity, hi = -Infinity;
          p.history.forEach(function (v) { if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
          p.lower.forEach(function (v) { lo = Math.min(lo, v); }); p.upper.forEach(function (v) { hi = Math.max(hi, v); });
          var pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
          function X(i) { return M.l + i / Math.max(N - 1, 1) * iw; } function Y(v) { return M.t + (1 - (v - lo) / (hi - lo)) * ih; }
          var yt = ticks(lo, hi, 4), ystep = tickStep(yt);
          yt.forEach(function (v) { svgEl.appendChild(s("line", { x1: M.l, x2: M.l + iw, y1: Y(v), y2: Y(v), stroke: "var(--grid)", "stroke-width": 1 })); var t = s("text", { x: M.l - 8, y: Y(v) + 3.5, "text-anchor": "end" }); t.textContent = fmtTick(v, ystep); svgEl.appendChild(t); });
          svgEl.appendChild(s("line", { x1: M.l, x2: M.l + iw, y1: M.t + ih, y2: M.t + ih, stroke: "var(--axis)", "stroke-width": 1 }));
          var sameDay = labels[0].slice(0, 10) === labels[N - 1].slice(0, 10);
          var marks = [[0, "start"], [N - 1, "end"]];
          if (nh > 1 && X(N - 1) - X(nh - 1) > 90 && X(nh - 1) - X(0) > 90) marks.push([nh - 1, "middle"]);
          marks.forEach(function (mk) { var t = s("text", { x: X(mk[0]), y: H - 8, "text-anchor": mk[1] }); t.textContent = shortTime(labels[mk[0]], sameDay); svgEl.appendChild(t); });
          var up = "", dn = "";
          for (var k = 0; k < F.steps; k++) { up += (up ? "L" : "M") + X(nh + k).toFixed(1) + "," + Y(p.upper[k]).toFixed(1); dn = "L" + X(nh + k).toFixed(1) + "," + Y(p.lower[k]).toFixed(1) + dn; }
          svgEl.appendChild(s("path", { d: up + dn + "Z", fill: "var(--series)", "fill-opacity": 0.16 }));
          function path(vals, color, w) { var d = "", pen = false; for (var i = 0; i < N; i++) { if (vals[i] === null) { pen = false; continue; } d += (pen ? "L" : "M") + X(i).toFixed(1) + "," + Y(vals[i]).toFixed(1); pen = true; } svgEl.appendChild(s("path", { d: d, fill: "none", stroke: color, "stroke-width": w, "stroke-linejoin": "round", "stroke-linecap": "round" })); }
          path(hist, "var(--muted)", 1.5); path(fc, "var(--series)", 2);
          if (nh) { svgEl.appendChild(s("line", { x1: X(nh - 1), x2: X(nh - 1), y1: M.t, y2: M.t + ih, stroke: "var(--axis)", "stroke-width": 1 })); var nt = s("text", { x: X(nh - 1) - 5, y: M.t + 10, "text-anchor": "end", class: "strong halo" }); nt.textContent = "forecast starts"; svgEl.appendChild(nt); }
          var end = s("circle", { cx: X(N - 1), cy: Y(p.values[F.steps - 1]), r: 4, fill: "var(--series)", stroke: "var(--surface)", "stroke-width": 2 }); svgEl.appendChild(end);
          var cross = s("line", { y1: M.t, y2: M.t + ih, stroke: "var(--ink2)", "stroke-width": 1, visibility: "hidden" }); svgEl.appendChild(cross);
          var hit = s("rect", { x: M.l, y: M.t, width: iw, height: ih, fill: "transparent", tabindex: 0, "aria-label": p.name + " forecast. Use arrow keys to read values." }); svgEl.appendChild(hit);
          var cur = -1;
          function at(i, evt) {
            cur = Math.max(0, Math.min(N - 1, i)); cross.setAttribute("x1", X(cur)); cross.setAttribute("x2", X(cur)); cross.setAttribute("visibility", "visible");
            if (!evt) { var b = svgEl.getBoundingClientRect(); evt = { clientX: b.left + X(cur) * b.width / W, clientY: b.top + M.t + 8 }; }
            var rows = cur < nh ? [{ label: "Observed", value: fmt(p.history[cur]), key: "expected" }] : [{ label: "Forecast", value: fmt(p.values[cur - nh]), key: "actual" }, { label: "80% interval", value: fmt(p.lower[cur - nh]) + " to " + fmt(p.upper[cur - nh]) }];
            showTip(evt, labels[cur] || ("step " + cur), rows);
          }
          function leave() { cross.setAttribute("visibility", "hidden"); hideTip(); }
          hit.addEventListener("pointermove", function (evt) { var b = svgEl.getBoundingClientRect(); at(Math.round(((evt.clientX - b.left) * W / b.width - M.l) / iw * (N - 1)), evt); });
          hit.addEventListener("pointerleave", leave); hit.addEventListener("blur", leave);
          hit.addEventListener("focus", function () { at(cur < 0 ? nh : cur); });
          hit.addEventListener("keydown", function (evt) { if (evt.key === "ArrowRight") { at(cur + 1); evt.preventDefault(); } else if (evt.key === "ArrowLeft") { at(cur - 1); evt.preventDefault(); } });
        }
        charts.push(render); render();
      })();
    });
    var fn = document.getElementById("forecastnotes"); F.notes.forEach(function (t) { fn.appendChild(h("li", null, t)); });
  }

  /* ---------- trust table & notes ---------- */
  var M0 = D.model, trust = "";
  if (M0.train_rows) trust = "Trained on " + M0.train_rows.toLocaleString() + " rows and calibrated on " + (M0.validation_rows ? "the " + M0.validation_rows.toLocaleString() + " rows that followed" : "the same rows (log too short to hold data back)") + ". ";
  trust += "Skill is the reduction in one-step prediction error compared with assuming the next reading equals the current one; 0% means the model adds nothing for that sensor.";
  document.getElementById("trustsub").textContent = trust;
  var skillBody = document.querySelector("#skill tbody"), sensorSkill = M0.sensor_skill || {}, byName = {};
  D.ranking.forEach(function (r) { byName[r.name] = r; });
  D.sensors.forEach(function (name) {
    var tr = h("tr"), v = sensorSkill[name]; tr.appendChild(h("td", null, name));
    var cell = h("td"); if (v !== undefined && v !== null) { var m = h("span", "meter"), fill = h("i"); fill.style.width = Math.max(0, Math.min(1, v)) * 100 + "%"; m.appendChild(fill); cell.appendChild(m); cell.appendChild(document.createTextNode(pct(v))); } else cell.textContent = "n/a (constant in training)";
    tr.appendChild(cell);
    var r = byName[name] || { steps: 0, peak: null };
    tr.appendChild(h("td", "num", r.steps.toLocaleString())); tr.appendChild(h("td", "num", fmt(r.peak))); skillBody.appendChild(tr);
  });
  var notes = document.getElementById("notes");
  (M0.warnings || []).concat(D.notes, M0.notes || []).forEach(function (t) { notes.appendChild(h("li", null, t)); });
  document.getElementById("footer").textContent = "Generated " + D.generated + " by RCWE " + D.version + " · sensitivity threshold " + fmt(D.threshold) + " sigma · " + D.mode + " mode"
    + (M0.device ? " · model trained on " + M0.device : "");

  /* ---------- theme & resize ---------- */
  document.getElementById("theme").addEventListener("click", function () {
    var root = document.documentElement, dark = root.dataset.theme ? root.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
  });
  var pending = null;
  function redrawAll() { clearTimeout(pending); pending = setTimeout(function () { charts.forEach(function (draw) { draw(); }); }, 60); }
  charts.forEach(function (draw) { draw(); });            /* panels were laid out one by one: draw again at final widths */
  window.addEventListener("resize", redrawAll);
})();
</script>
</body>
</html>
"""
