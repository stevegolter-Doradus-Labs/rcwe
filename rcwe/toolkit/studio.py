"""
RCWE Studio: the toolkit in a browser, served from your own machine.

    rcwe serve            ->  http://127.0.0.1:8765            upload a log, get the report
    rcwe ui               ->  http://127.0.0.1:8766/mission-control   live engine dashboard

Nothing leaves the computer: files are posted to this local server, analysed by the same
`TelemetryModel` the command line uses, and the results are written to the output folder.

The server binds to 127.0.0.1 by default and has no login. It refuses requests whose Host
or Origin header does not match its own address, so a web page open in another tab cannot
drive it. Bind it to another interface only on a network you trust.
"""
from __future__ import annotations

import json
import queue
import re
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from rcwe.paths import output_dir

__all__ = ["serve", "create_server", "JobRunner", "STUDIO_HTML"]

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_JOB_ID = re.compile(r"^[0-9a-f]{16}$")
_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson"}
_DOWNLOADS = {
    "report": ("report.html", "text/html; charset=utf-8", False),
    "result.json": ("result.json", "application/json", True),
    "scores.csv": ("scores.csv", "text/csv; charset=utf-8", True),
    "forecast.csv": ("forecast.csv", "text/csv; charset=utf-8", True),
    "model.rcwe": ("model.rcwe", "application/octet-stream", True),
}


class _BadRequest(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------------------
# analysis jobs
# --------------------------------------------------------------------------------------
class JobRunner:
    """Runs analyses one at a time on a worker thread and keeps their status for polling."""

    def __init__(self, root: Optional[Path] = None, keep: int = 25):
        self.root = Path(root) if root else output_dir() / "studio"
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()
        self._worker = threading.Thread(target=self._work, name="rcwe-studio-worker", daemon=True)
        self._worker.start()

    # ---- public ------------------------------------------------------------------------
    def submit(self, spec: Dict[str, Any]) -> str:
        spec = self._validate(spec)
        job_id = secrets.token_hex(8)
        folder = self.root / job_id
        folder.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job_id] = {
                "job": job_id, "name": spec["name"], "status": "queued", "stage": "queued", "fraction": 0.0,
                "message": "Waiting for the previous analysis to finish" if self._queue.qsize() else "Starting",
                "created": time.time(), "folder": str(folder), "files": [],
            }
            self._order.append(job_id)
            for stale in self._order[:-self.keep]:
                self._jobs.pop(stale, None)
            self._order = self._order[-self.keep:]
        self._queue.put((job_id, spec))
        return job_id

    def status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else {k: v for k, v in job.items() if k != "folder"}

    def recent(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [{k: self._jobs[j].get(k) for k in ("job", "name", "status", "headline", "events", "created")}
                    for j in reversed(self._order) if j in self._jobs]

    def file(self, job_id: str, key: str) -> Optional[Path]:
        if not _JOB_ID.match(job_id) or key not in _DOWNLOADS:
            return None
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job["status"] != "done":
            return None
        path = Path(job["folder"]) / _DOWNLOADS[key][0]
        return path if path.is_file() else None

    # ---- internals -----------------------------------------------------------------------
    @staticmethod
    def _validate(spec: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(spec, dict):
            raise _BadRequest(400, "The request body must be a JSON object.")
        demo = bool(spec.get("demo"))
        text, train_text = spec.get("text"), spec.get("train_text")
        if not demo and (not isinstance(text, str) or not text.strip()):
            raise _BadRequest(400, "No file content received. Choose a telemetry file (or use the demo).")
        if train_text is not None and not isinstance(train_text, str):
            raise _BadRequest(400, "train_text must be text.")
        sensitivity = spec.get("sensitivity", "medium")
        if isinstance(sensitivity, str) and sensitivity not in ("low", "medium", "high"):
            try:
                sensitivity = float(sensitivity)
            except ValueError:
                raise _BadRequest(400, "sensitivity must be low, medium, high or a number.")
        if not isinstance(sensitivity, str) and not 0.5 <= float(sensitivity) <= 100:
            raise _BadRequest(400, "A numeric sensitivity must be between 0.5 and 100 sigmas.")
        try:
            fraction = float(spec.get("train_fraction", 0.6))
            forecast = int(spec.get("forecast", 0) or 0)
        except (TypeError, ValueError):
            raise _BadRequest(400, "train_fraction and forecast must be numbers.")
        if not 0.1 <= fraction <= 0.95:
            raise _BadRequest(400, "train_fraction must be between 0.1 and 0.95.")
        if not 0 <= forecast <= 2000:
            raise _BadRequest(400, "forecast must be between 0 and 2000 steps.")

        def suffix(name: Any) -> str:
            ext = Path(str(name or "")).suffix.lower()
            return ext if ext in _SUFFIXES else ".csv"

        name = str(spec.get("name") or ("pump station demo" if demo else "upload"))[:200]
        return {
            "demo": demo, "name": name, "text": text, "suffix": suffix(spec.get("name")),
            "train_text": train_text if (train_text and train_text.strip()) else None,
            "train_suffix": suffix(spec.get("train_name")), "train_name": str(spec.get("train_name") or "")[:200],
            "sensitivity": sensitivity, "train_fraction": fraction, "forecast": forecast, "adapt": bool(spec.get("adapt")),
        }

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def _work(self) -> None:
        while True:
            job_id, spec = self._queue.get()
            try:
                self._run(job_id, spec)
            except Exception as exc:  # noqa: BLE001 - a failed job must never take the worker down
                from rcwe.toolkit import DataError
                message = str(exc) if isinstance(exc, (DataError, ValueError)) else f"{type(exc).__name__}: {exc}"
                self._update(job_id, status="error", error=message, message="Failed")
            finally:
                self._queue.task_done()

    def _run(self, job_id: str, spec: Dict[str, Any]) -> None:
        from rcwe.toolkit import TelemetryModel, load_table
        from rcwe.toolkit.demo import split_demo

        folder = Path(self._jobs[job_id]["folder"])
        self._update(job_id, status="running", stage="prepare", message="Reading the data")

        def progress(stage: str, fraction: float, message: str) -> None:
            overall = fraction * 0.75 if stage in ("prepare", "train", "calibrate", "done") else 0.75 + 0.2 * fraction
            self._update(job_id, stage=stage, fraction=round(min(overall, 0.97), 3), message=message)

        if spec["demo"]:
            train_source, table, _ = split_demo()
            learned_from = "ten days of healthy operation (built-in demo)"
        else:
            # Uploaded text is written under a name this server chose, then parsed from that path:
            # the content can never be mistaken for a file name.
            upload = folder / f"upload{spec['suffix']}"
            upload.write_text(spec["text"], encoding="utf-8")
            table = load_table(upload)
            table.source = spec["name"]
            if spec["train_text"]:
                train_file = folder / f"train{spec['train_suffix']}"
                train_file.write_text(spec["train_text"], encoding="utf-8")
                train_source = load_table(train_file)
                train_source.source = spec["train_name"] or "training log"
                learned_from = f"{spec['train_name'] or 'the training log'} ({train_source.n_rows:,} rows)"
            else:
                rows = max(12, min(table.n_rows, int(round(table.n_rows * spec["train_fraction"]))))
                train_source = table.slice(0, rows)
                learned_from = f"the first {rows:,} of {table.n_rows:,} rows"

        model = TelemetryModel().fit(train_source, progress=progress)
        result = model.detect(table, sensitivity=spec["sensitivity"], adapt=spec["adapt"], progress=progress)
        files = ["report", "result.json", "scores.csv", "model.rcwe"]
        if spec["forecast"]:
            result.forecast = model.forecast(spec["forecast"], data=table)
            result.forecast.to_csv(folder / "forecast.csv")
            files.append("forecast.csv")
        self._update(job_id, stage="report", fraction=0.98, message="Writing the report")
        result.to_html(folder / "report.html", title=f"RCWE analysis: {spec['name']}")
        result.to_json(folder / "result.json", include_series=False)
        result.to_csv(folder / "scores.csv")
        model.save(folder / "model.rcwe")

        count = result.n_events
        headline = (f"{count} anomaly event{'s' if count != 1 else ''} in {result.n_rows:,} rows"
                    if count else f"no anomalies detected in {result.n_rows:,} rows")
        self._update(
            job_id, status="done", stage="done", fraction=1.0, message="Done", headline=headline, events=count,
            event_lines=[e.description for e in result.events[:20]], summary=result.summary(),
            model_summary=model.summary(), learned_from=learned_from, files=files,
            notes=list(model.fit_summary_.warnings) + list(result.notes),
        )


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------
class _State:
    """Shared by every request handler of one server."""

    def __init__(self, jobs_root: Optional[Path] = None):
        self.jobs = JobRunner(jobs_root)
        self._engine: Any = None
        self._engine_lock = threading.Lock()

    def engine(self) -> Any:
        with self._engine_lock:
            if self._engine is None:
                from rcwe.ui.mission_control import MissionControlEngine
                self._engine = MissionControlEngine()
            return self._engine


def _make_handler(state: _State) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "RCWEStudio"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature fixed by the base class
            return

        # ---- plumbing ------------------------------------------------------------------
        def _send(self, status: int, body: bytes, content_type: str, extra: Optional[Dict[str, str]] = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _html(self, text: str) -> None:
            self._send(200, text.encode("utf-8"), "text/html; charset=utf-8")

        def _guard(self, writing: bool) -> None:
            """Only talk to pages that this server itself served (blocks cross-site and DNS-rebinding requests)."""
            host = (self.headers.get("Host") or "").strip().lower()
            bound = self.server.server_address[0]
            port = self.server.server_address[1]
            allowed = {f"{bound}:{port}", f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
            if bound in ("127.0.0.1", "::1", "localhost") and host not in allowed:
                raise _BadRequest(403, "Request refused: unexpected Host header.")
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc.lower() != host:
                raise _BadRequest(403, "Request refused: it came from a different site.")
            if writing and not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
                raise _BadRequest(415, "Send JSON (Content-Type: application/json).")

        def _body(self) -> Any:
            try:
                length = int(self.headers.get("Content-Length") or "")
            except ValueError:
                raise _BadRequest(411, "Content-Length is required.")
            if length > MAX_UPLOAD_BYTES + 1024 * 1024:
                raise _BadRequest(413, f"Upload too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB). Use `rcwe analyze` for bigger files.")
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise _BadRequest(400, "The request body is not valid JSON.")

        def _dispatch(self, writing: bool) -> None:
            try:
                self._guard(writing)
                url = urlparse(self.path)
                query = {k: v[-1] for k, v in parse_qs(url.query).items()}
                (self._post if writing else self._get)(url.path.rstrip("/") or "/", query)
            except _BadRequest as exc:
                self._json({"error": str(exc)}, exc.status)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # noqa: BLE001
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch(False)

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch(True)

        # ---- routes --------------------------------------------------------------------
        def _get(self, path: str, query: Dict[str, str]) -> None:
            if path == "/":
                from rcwe import __version__
                return self._html(STUDIO_HTML.replace("__VERSION__", __version__))
            if path == "/mission-control":
                from rcwe.ui.mission_control import HTML_TEMPLATE
                return self._html(HTML_TEMPLATE)
            if path == "/api/health":
                from rcwe import __version__
                return self._json({"ok": True, "version": __version__})
            if path == "/api/jobs":
                return self._json({"jobs": state.jobs.recent()})
            if path in ("/api/v10/state", "/api/state"):
                return self._json(state.engine().snapshot(self._since(query)))
            if path == "/api/v10/tick":
                return self._json(state.engine().tick(self._since(query)))
            parts = path.strip("/").split("/")
            if len(parts) >= 3 and parts[:2] == ["api", "jobs"] and _JOB_ID.match(parts[2]):
                if len(parts) == 3:
                    job = state.jobs.status(parts[2])
                    if job is None:
                        raise _BadRequest(404, "Unknown analysis job.")
                    return self._json(job)
                if len(parts) == 4:
                    target = state.jobs.file(parts[2], parts[3])
                    if target is None:
                        raise _BadRequest(404, "That result is not available.")
                    name, content_type, attach = _DOWNLOADS[parts[3]]
                    extra = {"Content-Disposition": f'attachment; filename="{name}"'} if attach else None
                    return self._send(200, target.read_bytes(), content_type, extra)
            raise _BadRequest(404, "Not found.")

        def _post(self, path: str, query: Dict[str, str]) -> None:
            body = self._body()
            if path == "/api/analyze":
                return self._json({"job": state.jobs.submit(body)}, 202)
            if path in ("/api/v10/action", "/api/action"):
                def integer(key: str) -> Optional[int]:
                    try:
                        return int(query[key]) if key in query else None
                    except ValueError:
                        raise _BadRequest(400, f"{key} must be an integer.")
                result = state.engine().trigger_action(query.get("type", "step"), row=integer("r"), col=integer("c"))
                return self._json(result, 200 if result.get("status") == "ok" else 400)
            raise _BadRequest(404, "Not found.")

        @staticmethod
        def _since(query: Dict[str, str]) -> int:
            try:
                return int(query.get("since", 0))
            except ValueError:
                return 0

    return Handler


def create_server(host: str = "127.0.0.1", port: int = 8765, jobs_root: Optional[Path] = None) -> ThreadingHTTPServer:
    """Build the server without starting it (use `serve_forever()` on the result)."""
    server = ThreadingHTTPServer((host, port), _make_handler(_State(jobs_root)))
    server.daemon_threads = True
    return server


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True, start_page: str = "/") -> None:
    """Run RCWE Studio until interrupted."""
    server = None
    for candidate in range(port, port + 20):
        try:
            server = create_server(host, candidate)
            break
        except OSError:
            continue
    if server is None:
        raise OSError(f"No free port between {port} and {port + 19}. Pass --port to choose another.")
    actual = server.server_address[1]
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    base = f"http://{shown}:{actual}"
    print("RCWE Studio is running")
    print(f"  Upload and analyse a log   {base}/")
    print(f"  Mission Control (live)     {base}/mission-control")
    if actual != port:
        print(f"  (port {port} was busy, using {actual})")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("  WARNING: bound to a network interface with no login. Anyone who can reach this address can use it.")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(base + start_page)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping RCWE Studio.")
    finally:
        server.server_close()


STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RCWE Studio</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --border: rgba(11,11,11,0.10); --wash: rgba(11,11,11,0.04);
    --accent: #2a78d6; --good: #0ca30c; --critical: #d03b3b; --warning: #fab219;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --border: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.05); --accent: #3987e5;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 24px 20px 56px; }
  header { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; margin-bottom: 18px; }
  h1 { font-size: 22px; margin: 0; font-weight: 650; }
  h2 { font-size: 16px; margin: 0 0 4px; font-weight: 650; }
  a { color: var(--accent); }
  .sub { color: var(--ink2); font-size: 13px; }
  .hint { color: var(--muted); font-size: 12px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; margin-bottom: 16px; }
  .cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
  .drop { display: block; border: 1.5px dashed var(--muted); border-radius: 10px; padding: 22px 16px; text-align: center; cursor: pointer; color: var(--ink2); background: var(--wash); }
  .drop.over, .drop:hover { border-color: var(--accent); color: var(--ink); }
  .drop strong { display: block; color: var(--ink); font-size: 15px; margin-bottom: 2px; overflow-wrap: anywhere; }
  .drop input { display: none; }
  .opts { display: flex; flex-wrap: wrap; gap: 18px 28px; align-items: flex-end; margin-top: 14px; }
  label.field { display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: var(--ink2); }
  label.check { display: flex; gap: 8px; align-items: center; font-size: 13px; color: var(--ink2); }
  select, input[type=number] { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; min-width: 110px; }
  button { font: inherit; border-radius: 8px; padding: 9px 16px; cursor: pointer; border: 1px solid var(--border); background: var(--surface); color: var(--ink); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  button:disabled { opacity: 0.5; cursor: default; }
  button:focus-visible, select:focus-visible, input:focus-visible, .drop:focus-within { outline: 2px solid var(--accent); outline-offset: 2px; }
  .actions { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 16px; }
  .bar { height: 8px; border-radius: 4px; background: color-mix(in srgb, var(--accent) 18%, transparent); overflow: hidden; margin: 10px 0 6px; }
  .bar i { display: block; height: 100%; width: 0; background: var(--accent); border-radius: 4px; transition: width 0.3s ease; }
  .status { display: inline-flex; gap: 8px; align-items: center; font-weight: 600; font-size: 17px; }
  .dot { width: 12px; height: 12px; border-radius: 50%; flex: none; }
  pre { white-space: pre-wrap; overflow-wrap: anywhere; font: 12.5px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--ink2); margin: 10px 0 0; }
  .downloads { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
  .downloads a { text-decoration: none; border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; color: var(--ink); font-size: 13px; background: var(--surface); }
  .downloads a:hover { background: var(--wash); }
  iframe { width: 100%; height: 1500px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
  .error { color: var(--critical); font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  td, th { text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--grid); }
  th { color: var(--ink2); font-size: 12px; }
  [hidden] { display: none !important; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>RCWE Studio</h1>
      <div class="sub">Find and explain anomalies in telemetry. Everything runs on this computer; files are not sent anywhere else.</div>
    </div>
    <div class="sub">RCWE __VERSION__ · <a href="/mission-control">Mission Control</a></div>
  </header>

  <section class="card">
    <h2>1. Choose your telemetry</h2>
    <div class="sub">CSV, TSV, JSON or JSON-lines with one row per time step. Numeric columns become sensors; columns named like <code>act_*</code> or <code>*_setpoint</code> become controls.</div>
    <div class="cols" style="margin-top:12px">
      <label class="drop" id="dropScan">
        <strong id="scanName">Log to scan</strong>
        <span id="scanHint">Drop a file here or click to choose</span>
        <input type="file" id="scanFile" accept=".csv,.tsv,.txt,.json,.jsonl,.ndjson">
      </label>
      <label class="drop" id="dropTrain">
        <strong id="trainName">Known-good log (optional)</strong>
        <span id="trainHint">A period of normal operation to learn from. Without it, the first part of the log is used.</span>
        <input type="file" id="trainFile" accept=".csv,.tsv,.txt,.json,.jsonl,.ndjson">
      </label>
    </div>

    <h2 style="margin-top:20px">2. Options</h2>
    <div class="opts">
      <label class="field">Sensitivity
        <select id="sensitivity">
          <option value="low">Low: only strong anomalies</option>
          <option value="medium" selected>Medium (recommended)</option>
          <option value="high">High: flag subtle deviations</option>
        </select>
      </label>
      <label class="field" id="fractionField">Learn from the first
        <select id="fraction">
          <option value="0.3">30% of the log</option>
          <option value="0.5">50% of the log</option>
          <option value="0.6" selected>60% of the log</option>
          <option value="0.8">80% of the log</option>
        </select>
      </label>
      <label class="field">Forecast ahead (steps)
        <input type="number" id="forecast" min="0" max="2000" value="48">
      </label>
      <label class="check"><input type="checkbox" id="adapt"> Adaptive: flag when a change starts, then accept it as the new normal</label>
    </div>

    <div class="actions">
      <button class="primary" id="run" type="button" disabled>Analyse</button>
      <button id="demo" type="button">Try the built-in demo</button>
      <span class="hint" id="runHint">Choose a log to scan first.</span>
    </div>
  </section>

  <section class="card" id="progressCard" hidden aria-live="polite">
    <h2 id="progressTitle">Working…</h2>
    <div class="bar"><i id="progressBar"></i></div>
    <div class="sub" id="progressText"></div>
  </section>

  <section class="card" id="resultCard" hidden aria-live="polite">
    <div class="status"><span class="dot" id="resultDot"></span><span id="resultHeadline"></span></div>
    <div class="sub" id="resultSub"></div>
    <pre id="resultSummary"></pre>
    <div class="downloads" id="downloads"></div>
  </section>

  <iframe id="report" title="Analysis report" hidden></iframe>

  <section class="card" id="historyCard" hidden>
    <h2>Earlier analyses in this session</h2>
    <table><thead><tr><th>Log</th><th>Result</th><th></th></tr></thead><tbody id="history"></tbody></table>
  </section>
</div>
<script>
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var files = { scan: null, train: null }, running = false;

  function readFile(file) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () { resolve(reader.result); };
      reader.onerror = function () { reject(new Error("could not read " + file.name)); };
      reader.readAsText(file);
    });
  }
  function choose(kind, file) {
    if (!file) return;
    if (file.size > 64 * 1024 * 1024) { showError(file.name + " is larger than 64 MB. Use `rcwe analyze` on the command line for files this big."); return; }
    files[kind] = file;
    $(kind + "Name").textContent = file.name;
    $(kind + "Hint").textContent = (file.size / 1024).toFixed(0) + " KB · click to change";
    $("fractionField").hidden = !!files.train;
    $("run").disabled = !files.scan || running;
    $("runHint").textContent = files.scan ? "" : "Choose a log to scan first.";
  }
  ["scan", "train"].forEach(function (kind) {
    var zone = $("drop" + kind.charAt(0).toUpperCase() + kind.slice(1)), input = $(kind + "File");
    input.addEventListener("change", function () { choose(kind, input.files[0]); });
    zone.addEventListener("dragover", function (e) { e.preventDefault(); zone.classList.add("over"); });
    zone.addEventListener("dragleave", function () { zone.classList.remove("over"); });
    zone.addEventListener("drop", function (e) { e.preventDefault(); zone.classList.remove("over"); choose(kind, e.dataTransfer.files[0]); });
  });

  function setBusy(flag) {
    running = flag;
    $("run").disabled = flag || !files.scan;
    $("demo").disabled = flag;
  }
  function showError(message) {
    $("progressCard").hidden = false;
    $("progressTitle").textContent = "That did not work";
    $("progressTitle").className = "error";
    $("progressBar").style.width = "0";
    $("progressText").textContent = message;
    setBusy(false);
  }
  async function start(payload, label) {
    setBusy(true);
    $("resultCard").hidden = true; $("report").hidden = true;
    $("progressCard").hidden = false;
    $("progressTitle").className = ""; $("progressTitle").textContent = "Analysing " + label + "…";
    $("progressBar").style.width = "2%"; $("progressText").textContent = "Uploading";
    try {
      var res = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      var body = await res.json();
      if (!res.ok) throw new Error(body.error || "The server refused the request.");
      poll(body.job);
    } catch (err) { showError(err.message); }
  }
  async function poll(id) {
    try {
      var res = await fetch("/api/jobs/" + id, { cache: "no-store" });
      var job = await res.json();
      if (!res.ok) throw new Error(job.error || "The analysis was lost.");
      if (job.status === "error") { showError(job.error); refreshHistory(); return; }
      if (job.status === "done") { show(job); setBusy(false); refreshHistory(); return; }
      $("progressBar").style.width = Math.max(2, Math.round((job.fraction || 0) * 100)) + "%";
      $("progressText").textContent = job.message || "";
      setTimeout(function () { poll(id); }, 600);
    } catch (err) { showError(err.message); }
  }
  function show(job) {
    $("progressCard").hidden = true;
    $("resultCard").hidden = false;
    $("resultDot").style.background = job.events ? "var(--critical)" : "var(--good)";
    $("resultHeadline").textContent = job.name + ": " + job.headline;
    $("resultSub").textContent = "Normal behaviour was learned from " + job.learned_from + ".";
    $("resultSummary").textContent = job.model_summary + "\n\n" + job.summary + (job.notes && job.notes.length ? "\n\n" + job.notes.map(function (n) { return "Note: " + n; }).join("\n") : "");
    var box = $("downloads"); box.textContent = "";
    var labels = { "report": "Open report in a new tab", "result.json": "Events (JSON)", "scores.csv": "Per-row scores (CSV)", "forecast.csv": "Forecast (CSV)", "model.rcwe": "Model file" };
    job.files.forEach(function (key) {
      var a = document.createElement("a");
      a.href = "/api/jobs/" + job.job + "/" + key; a.textContent = labels[key] || key;
      if (key === "report") { a.target = "_blank"; a.rel = "noopener"; }
      box.appendChild(a);
    });
    var frame = $("report");
    frame.src = "/api/jobs/" + job.job + "/report"; frame.hidden = false;
    $("resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  async function refreshHistory() {
    try {
      var res = await fetch("/api/jobs", { cache: "no-store" }), data = await res.json(), body = $("history");
      body.textContent = "";
      data.jobs.filter(function (j) { return j.status === "done"; }).forEach(function (j) {
        var tr = document.createElement("tr"), name = document.createElement("td"), what = document.createElement("td"), link = document.createElement("td"), a = document.createElement("a");
        name.textContent = j.name; what.textContent = j.headline;
        a.href = "/api/jobs/" + j.job + "/report"; a.target = "_blank"; a.rel = "noopener"; a.textContent = "Report";
        link.appendChild(a); tr.appendChild(name); tr.appendChild(what); tr.appendChild(link); body.appendChild(tr);
      });
      $("historyCard").hidden = body.children.length < 2;
    } catch (err) { /* history is optional */ }
  }
  function options() {
    return { sensitivity: $("sensitivity").value, train_fraction: parseFloat($("fraction").value), forecast: parseInt($("forecast").value || "0", 10) || 0, adapt: $("adapt").checked };
  }
  $("run").addEventListener("click", async function () {
    if (!files.scan || running) return;
    try {
      var payload = options();
      payload.name = files.scan.name; payload.text = await readFile(files.scan);
      if (files.train) { payload.train_name = files.train.name; payload.train_text = await readFile(files.train); }
      start(payload, files.scan.name);
    } catch (err) { showError(err.message); }
  });
  $("demo").addEventListener("click", function () {
    if (running) return;
    var payload = options(); payload.demo = true;
    start(payload, "the pump station demo");
  });
  refreshHistory();
})();
</script>
</body>
</html>
"""
