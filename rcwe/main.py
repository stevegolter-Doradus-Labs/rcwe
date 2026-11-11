"""
The `rcwe` command.

    rcwe quickstart                         # 60-second tour on built-in demo data
    rcwe analyze my_log.csv                 # one file in, explained anomalies + HTML report out
    rcwe fit normal.csv -o plant.rcwe       # learn normal behaviour once ...
    rcwe detect today.csv -m plant.rcwe     # ... and reuse it on new logs
    rcwe forecast -m plant.rcwe --steps 48
    rcwe watch live.csv -m plant.rcwe       # follow a growing file and raise alerts
    rcwe serve                              # the same workflow in a browser
    rcwe doctor                             # check the installation

Run `rcwe <command> --help` for the options of each command.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

EXIT_OK, EXIT_ERROR, EXIT_ANOMALY = 0, 1, 2


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
class _Progress:
    """Single-line progress on a terminal; an occasional line when output is piped."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.tty = sys.stderr.isatty()
        self._last = 0.0
        self._open = False

    def __call__(self, stage: str, fraction: float, message: str) -> None:
        if not self.enabled:
            return
        if self.tty:
            sys.stderr.write("\r  " + message[:74].ljust(74))
            sys.stderr.flush()
            self._open = True
            if fraction >= 1.0:
                self.close()
        elif fraction >= 1.0 or time.monotonic() - self._last > 5.0:
            self._last = time.monotonic()
            print(f"  {message}", file=sys.stderr)

    def close(self) -> None:
        if self._open:
            sys.stderr.write("\r" + " " * 78 + "\r")
            sys.stderr.flush()
            self._open = False


def _split(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _sensitivity(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value


def _show(path: Any) -> str:
    """A path as the user would type it: relative when it is inside the current folder."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _quote(path: Any) -> str:
    """A path as it can be pasted into a shell command."""
    text = _show(path)
    if os.name == "nt":
        return f'"{text}"' if any(ch in text for ch in " &()^") else text
    return shlex.quote(text)


def _file_target(path: Optional[str], option: str) -> Optional[str]:
    """Refuse, before any work is done, an output file name that is really a folder."""
    if path and Path(path).is_dir():
        from rcwe.toolkit import DataError
        raise DataError(f"{option} {path} is a folder; give a file name.")
    return path


def _folder_target(path: Any, option: str) -> Path:
    """Refuse, before any work is done, an output folder name that is really a file."""
    target = Path(path)
    if target.exists() and not target.is_dir():
        from rcwe.toolkit import DataError
        raise DataError(f"{option} {path} is a file; give a folder.")
    return target


def _number(kind: Callable[[str], Any], low: Optional[float] = None, high: Optional[float] = None,
            open_ends: bool = False) -> Callable[[str], Any]:
    """argparse type: a number within [low, high] (or strictly inside it when `open_ends`)."""
    def parse(text: str) -> Any:
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{text}' is not a {'whole ' if kind is int else ''}number") from None
        too_low = low is not None and (value <= low if open_ends else value < low)
        too_high = high is not None and (value >= high if open_ends else value > high)
        if too_low or too_high or value != value:
            if low is not None and high is not None:
                wanted = f"between {low:g} and {high:g}" + (" (exclusive)" if open_ends else "")
            else:
                wanted = f"{'greater than' if open_ends else 'at least'} {low:g}" if low is not None else f"at most {high:g}"
            raise argparse.ArgumentTypeError(f"must be {wanted}, got {text}")
        return value
    return parse


def _open_in_browser(path: str) -> None:
    try:
        webbrowser.open(Path(path).resolve().as_uri())
    except Exception:  # noqa: BLE001 - never fail a command because no browser is available
        pass


def _schema_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "sensors": _split(getattr(args, "sensors", None)),
        "controls": _split(getattr(args, "controls", None)),
        "time_column": getattr(args, "time_column", None),
        "label_column": getattr(args, "label_column", None),
    }


def _model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"seed": args.seed, "device": args.device}
    if getattr(args, "steps", None):
        kwargs["max_steps"] = args.steps
    return kwargs


def _add_schema_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("columns (all optional: detected automatically)")
    group.add_argument("--sensors", metavar="A,B,C", help="columns to model (default: every numeric column)")
    group.add_argument("--controls", metavar="X,Y", help="control/setpoint columns (default: act_*, u_*, ctrl_*, cmd_*, *_setpoint)")
    group.add_argument("--time-column", metavar="NAME", help="timestamp column")
    group.add_argument("--label-column", metavar="NAME", help="ground-truth anomaly labels, used only to score the result")


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training")
    group.add_argument("--steps", type=_number(int, 1), metavar="N", help="maximum training steps (default 400; stops early when converged)")
    group.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    group.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="where to train (default: auto)")


def _add_detect_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("detection")
    group.add_argument("--sensitivity", default="medium", metavar="LEVEL",
                       help="low | medium | high, or a threshold in sigmas (default: medium = 4.5)")
    group.add_argument("--adapt", action="store_true",
                       help="let the model keep learning during the scan: flags the onset of changes, then accepts the new normal")
    group.add_argument("--no-range-check", action="store_true", help="do not flag readings merely for being outside the training range")
    group.add_argument("--forecast", type=_number(int, 0), metavar="N", default=0, help="also forecast N steps past the end of the log")


def _print_events_hint(result: Any) -> None:
    """Where the deviations lasted longest. Each event above already names the sensor it started on."""
    ranked = sorted((r for r in result.sensor_ranking() if r["anomalous_steps"]),
                    key=lambda r: (r["anomalous_steps"], r["involvement"]), reverse=True)[:3]
    if ranked:
        print("\nMost time above threshold: "
              + ", ".join(f"{r['name']} ({r['anomalous_steps']:,} step{'' if r['anomalous_steps'] == 1 else 's'})" for r in ranked))


def _detect_and_export(model: Any, source: Any, args: argparse.Namespace, out_dir: Optional[Path], progress: _Progress,
                       title: Optional[str] = None) -> Any:
    for option in ("report", "json", "csv"):
        _file_target(getattr(args, option, None), f"--{option}")
    result = model.detect(
        source, sensitivity=_sensitivity(args.sensitivity), adapt=args.adapt,
        range_check=False if args.no_range_check else None, progress=progress,
    )
    progress.close()
    if args.forecast:
        result.forecast = model.forecast(args.forecast, data=source)
    print(result.summary())
    _print_events_hint(result)

    written: List[str] = []
    report = getattr(args, "report", None) or (str(out_dir / "report.html") if out_dir else None)
    as_json = getattr(args, "json", None) or (str(out_dir / "result.json") if out_dir else None)
    as_csv = getattr(args, "csv", None) or (str(out_dir / "scores.csv") if out_dir else None)
    if report:
        written.append("report   " + _show(result.to_html(report, title=title)))
    if as_json:
        written.append("json     " + _show(result.to_json(as_json, include_series=False)))
    if as_csv:
        written.append("scores   " + _show(result.to_csv(as_csv)))
    if result.forecast is not None:
        print("\n" + result.forecast.summary())
        if out_dir:
            written.append("forecast " + _show(result.forecast.to_csv(out_dir / "forecast.csv")))
    if written:
        print("\nWrote:")
        for line in written:
            print("  " + line)
    if report and getattr(args, "open", False):
        _open_in_browser(report)
    return result


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------
def cmd_quickstart(args: argparse.Namespace) -> int:
    from rcwe.toolkit import TelemetryModel
    from rcwe.toolkit.demo import write_demo_files

    out = _folder_target(args.out, "--out")
    progress = _Progress(not args.quiet)
    print("RCWE quickstart: a pump station with six sensors and two operator controls.\n")
    paths = write_demo_files(out, seed=args.seed)
    print(f"1/4  Demo data written to {_show(out)}/")
    print("       pump_station_normal.csv    ten days of healthy operation")
    print("       pump_station_incident.csv  the next four days, with five planted faults\n")

    print("2/4  Learning normal behaviour from pump_station_normal.csv ...")
    model = TelemetryModel(seed=args.seed, device=args.device).fit(paths["normal"], progress=progress)
    progress.close()
    print("     " + model.summary().replace("\n", "\n     ") + "\n")

    print("3/4  Scanning pump_station_incident.csv ...")
    result = model.detect(paths["incident"], progress=progress)
    progress.close()
    result.forecast = model.forecast(48, data=paths["incident"])
    print("     " + result.summary().replace("\n", "\n     ") + "\n")

    report = result.to_html(out / "report.html", title="RCWE quickstart: pump station incident scan")
    model_path = model.save(out / "pump_station.rcwe")
    result.to_json(out / "result.json", include_series=False)
    print("4/4  Results")
    print(f"       report  {_show(report)}")
    print(f"       model   {_show(model_path)}")
    print("\nNext steps")
    print("  Use your own data  rcwe analyze path/to/your_log.csv")
    print(f"  Reuse this model   rcwe detect {_quote(paths['incident'])} -m {_quote(model_path)} --report again.html")
    print("  Work in a browser  rcwe serve")
    if not args.no_open:
        _open_in_browser(report)
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    from rcwe.toolkit.doctor import format_checks, run_checks

    text, healthy = format_checks(run_checks(debug=args.debug))
    print(text)
    return EXIT_OK if healthy else EXIT_ERROR


def cmd_inspect(args: argparse.Namespace) -> int:
    from rcwe.toolkit import load_table

    table = load_table(args.file, **_schema_kwargs(args))
    info = table.describe()
    if args.json:
        print(json.dumps(info, indent=2))
        return EXIT_OK
    interval = info["sample_interval_seconds"]
    print(f"{info['source']}")
    print(f"  rows        {info['rows']:,}")
    print(f"  time        {info['time_column'] or '(none found: rows are treated as evenly spaced steps)'}"
          + (f"   {info['start']} to {info['end']}" if info["time_column"] else "")
          + (f"   every {interval:g}s" if interval else ""))
    print(f"  sensors     {len(info['sensors'])}")
    width = max((len(s["name"]) for s in info["sensor_stats"]), default=4)
    print(f"    {'name'.ljust(width)}  {'min':>12} {'mean':>12} {'max':>12} {'missing':>8}")
    for stat in info["sensor_stats"]:
        print(f"    {stat['name'].ljust(width)}  {stat['min']:>12.5g} {stat['mean']:>12.5g} {stat['max']:>12.5g} {stat['missing']:>8}")
    print(f"  controls    {', '.join(info['controls']) or '(none)'}")
    if info["label_column"]:
        print(f"  labels      {info['label_column']} ({info['labeled_anomalous_rows']:,} rows marked anomalous)")
    if info["ignored_columns"]:
        print(f"  ignored     {', '.join(info['ignored_columns'])}")
    for note in info["notes"]:
        print(f"  note        {note}")
    if info["rows"] < 80:
        print("  note        Fewer than 80 rows: models trained on this log are indicative only.")
    return EXIT_OK


def cmd_fit(args: argparse.Namespace) -> int:
    from rcwe.toolkit import TelemetryModel

    progress = _Progress(not args.quiet)
    out = _file_target(args.output or str(Path(args.file).with_suffix(".rcwe").name), "-o")
    model = TelemetryModel(**_model_kwargs(args)).fit(args.file, progress=progress, **_schema_kwargs(args))
    progress.close()
    print(model.summary())
    for note in model.fit_summary_.notes:
        print(f"Note: {note}")
    print(f"\nModel saved to {_show(model.save(out))}")
    print(f"Next: rcwe detect <new_log.csv> -m {_quote(out)} --report report.html")
    return EXIT_OK


def cmd_detect(args: argparse.Namespace) -> int:
    from rcwe.toolkit import TelemetryModel

    model = TelemetryModel.load(args.model, device=args.device)
    result = _detect_and_export(model, args.file, args, None, _Progress(not args.quiet), title=args.title)
    if args.fail_on_anomaly and result.n_events:
        return EXIT_ANOMALY
    return EXIT_OK


def cmd_forecast(args: argparse.Namespace) -> int:
    from rcwe.toolkit import TelemetryModel

    _file_target(args.csv, "--csv")
    _file_target(args.json, "--json")
    model = TelemetryModel.load(args.model, device=args.device)
    forecast = model.forecast(args.steps, data=args.file, future_controls=args.planned_controls)
    print(forecast.summary())
    if args.csv:
        print(f"\nWrote {_show(forecast.to_csv(args.csv))}")
    if args.json:
        print(f"Wrote {_show(forecast.to_json(args.json))}")
    return EXIT_OK


def cmd_analyze(args: argparse.Namespace) -> int:
    from rcwe.paths import output_dir
    from rcwe.toolkit import DataError, TelemetryModel, load_table

    progress = _Progress(not args.quiet)
    out = _folder_target(args.out, "--out") if args.out else output_dir() / Path(args.file).stem
    schema = _schema_kwargs(args)
    table = load_table(args.file, **schema)
    out.mkdir(parents=True, exist_ok=True)

    if args.train:
        try:
            train_source: Any = load_table(args.train, **schema)
        except DataError:
            if not schema["label_column"]:
                raise
            # A log of known-good operation has no reason to carry the label column.
            train_source = load_table(args.train, **dict(schema, label_column=None))
        print(f"Learning normal behaviour from {args.train} ...")
    else:
        rows = args.train_rows or int(round(table.n_rows * args.train_fraction))
        rows = max(12, min(rows, table.n_rows))
        train_source = table.slice(0, rows)
        share = f"the first {rows:,} of {table.n_rows:,} rows"
        print(f"Learning normal behaviour from {share} of {args.file} ...")
    model = TelemetryModel(**_model_kwargs(args)).fit(train_source, progress=progress)
    progress.close()
    print(model.summary())
    for note in model.fit_summary_.notes:
        print(f"Note: {note}")
    print()

    print(f"Scanning all {table.n_rows:,} rows ...")
    result = _detect_and_export(model, table, args, out, progress, title=args.title or f"RCWE analysis: {Path(args.file).name}")
    print("  model    " + _show(model.save(out / "model.rcwe")))
    if not args.train and result.n_events:
        print("\nTip: anomalies inside the rows used for training are partly learned as normal. If you have a log of "
              "known-good operation, pass it with --train for sharper results.")
    if args.fail_on_anomaly and result.n_events:
        return EXIT_ANOMALY
    return EXIT_OK


_DELIMITERS = (",", ";", "\t", "|")


def _follow_csv(path: Path, interval: float, once: bool,
                on_header: Optional[Callable[[List[str]], None]] = None) -> Iterator[Optional[Dict[str, str]]]:
    """
    Yield each row of a growing delimited file as a dict, and None once when the content that
    was already there has been read.

    Works like `tail -F`: a row is delivered when its line is complete, and if the file is
    truncated or replaced (log rotation) reading restarts from the top of the new content.
    The delimiter is taken from the header line (comma, semicolon, tab or pipe). Lines that
    start with "#" are comments. `on_header` is called with the column names each time a
    header line is read.
    """
    from rcwe.toolkit import DataError
    from rcwe.toolkit.table import _looks_like_data

    def identity(open_file: Any) -> Tuple[int, int]:
        info = os.fstat(open_file.fileno())
        return info.st_dev, info.st_ino

    handle = path.open("rb")
    try:
        ident = identity(handle)
        header: Optional[List[str]] = None
        earlier: Optional[List[str]] = None          # the header in force before a rotation
        delimiter = ","
        pending = b""
        caught_up = False
        while True:
            chunk = handle.readline()
            unfinished = False
            if chunk.endswith(b"\n"):
                line, pending = pending + chunk, b""
            else:
                pending += chunk                     # nothing new, or a row that is still being written
                if once and pending:
                    line, pending, unfinished = pending, b"", True     # the last line of a file has no newline
                else:
                    if not caught_up:
                        caught_up = True
                        yield None
                    if once:
                        return
                    time.sleep(interval)
                    try:
                        info = os.stat(path)
                    except OSError:
                        continue                     # mid-rotation: the file will be back
                    if (info.st_dev, info.st_ino) != ident or info.st_size < handle.tell():
                        handle.close()
                        handle = path.open("rb")
                        ident = identity(handle)
                        earlier, header, pending = header or earlier, None, b""
                    continue
            text = line.decode("utf-8", errors="replace").rstrip("\r\n").lstrip("\ufeff")
            if not text.strip() or text.lstrip().startswith("#"):
                continue
            if header is None:
                if earlier is not None and _looks_like_data(next(csv.reader([text], delimiter=delimiter))):
                    # Numbers where column names should be: after `copytruncate` rotation that is
                    # simply the next reading, and the columns are the ones the file had before.
                    header = earlier
                else:
                    delimiter = max(_DELIMITERS, key=text.count)
                    cells = next(csv.reader([text], delimiter=delimiter))
                    if _looks_like_data(cells):
                        raise DataError(f"{path} has no header row naming its columns.")
                    header = [cell.strip() for cell in cells]
                    if on_header is not None:
                        on_header(header)
                    continue
            cells = next(csv.reader([text], delimiter=delimiter))
            if unfinished and len(cells) != len(header):
                continue                             # cut off in the middle of being written
            yield dict(zip(header, cells))
    finally:
        handle.close()


def cmd_watch(args: argparse.Namespace) -> int:
    from rcwe.toolkit import DataError, TelemetryModel
    from rcwe.toolkit.table import _to_float

    model = TelemetryModel.load(args.model, device=args.device)
    monitor = model.stream(sensitivity=_sensitivity(args.sensitivity), adapt=args.adapt)
    path = Path(args.file)
    if not path.exists():
        raise DataError(f"File not found: {path}")
    log = open(_file_target(args.jsonl, "--jsonl"), "a", encoding="utf-8") if args.jsonl else None
    live = args.once or args.from_start
    print(f"Watching {path} with {args.model} (threshold {monitor.threshold:g} sigma, "
          f"{'adaptive' if args.adapt else 'reference'} mode). Press Ctrl+C to stop.")

    rows = alerts = quiet = 0
    event: Optional[Dict[str, Any]] = None

    def check_columns(columns: List[str]) -> None:
        absent = [name for name in model.sensors if name not in columns]
        if absent:
            raise DataError(
                f"{path} has no column named {', '.join(repr(name) for name in absent)}, which this model "
                f"was trained on. Columns found: {', '.join(name for name in columns if name) or '(none)'}."
            )

    def emit(kind: str, payload: Dict[str, Any]) -> None:
        if log:
            log.write(json.dumps({"type": kind, **payload}) + "\n")
            log.flush()

    def close_event() -> None:
        nonlocal event
        if event is not None:
            print(f"  cleared   {event['last_time']}  (lasted {event['rows']} row(s), peak {event['peak']:.1f} sigma)")
            emit("cleared", {"time": event["last_time"], "rows": event["rows"], "peak_score": round(event["peak"], 3)})
            event = None

    try:
        for raw in _follow_csv(path, args.interval, args.once, on_header=check_columns):
            if raw is None:
                if not live:
                    print(f"  primed on {rows:,} existing row(s); waiting for new ones ...")
                    live = True
                continue
            reading: Dict[str, Any] = {}
            for name in model.sensors + model.controls:
                value = _to_float(raw.get(name))
                reading[name] = None if value is None or value != value else value
            stamp = raw.get(model.time_column) if model.time_column else None
            try:
                step = monitor.update(reading, timestamp=stamp)
            except DataError as exc:
                print(f"  skipped   row {rows + 1}: {exc}")
                continue
            rows += 1
            if live and (step.alert or (step.is_anomaly and event is not None)):
                quiet = 0
                if event is None:
                    alerts += 1
                    lead = step.sensors[0]
                    print(f"  ANOMALY   {step.time}  score {step.score:.1f}  {lead['name']}: {lead['detail']}")
                    emit("anomaly", {"time": step.time, "score": round(step.score, 3), "sensors": step.sensors})
                    event = {"rows": 0, "peak": 0.0, "last_time": step.time}
                event["rows"] += 1
                event["peak"] = max(event["peak"], step.score)
                event["last_time"] = step.time
            elif event is not None:
                quiet += 1
                if quiet > 3:
                    close_event()
            if args.max_rows and rows >= args.max_rows:
                break
    except KeyboardInterrupt:
        print()
    finally:
        close_event()
        if log:
            log.close()
    print(f"Processed {rows:,} row(s); {alerts} alert(s).")
    return EXIT_ANOMALY if (args.fail_on_anomaly and alerts) else EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    from rcwe.toolkit.studio import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_open)
    return EXIT_OK


def cmd_ui(args: argparse.Namespace) -> int:
    from rcwe.ui.mission_control import MissionControlEngine, serve_mission_control

    if args.export:
        print(f"Exported standalone Mission Control page -> {MissionControlEngine.export_standalone_html(args.export)}")
        return EXIT_OK
    serve_mission_control(host=args.host, port=args.port, open_browser=not args.no_open)
    return EXIT_OK


def _bench_suites() -> Dict[str, Any]:
    def validate() -> None:
        from rcwe.cli import main
        main()

    def extreme() -> None:
        from rcwe.diagnostics.extreme_bench import main_extreme_and_symbolic
        main_extreme_and_symbolic()

    def dashboard() -> None:
        from rcwe.diagnostics.dashboard_v05 import main_v05
        main_v05()

    def realworld() -> None:
        from rcwe.benchmarks.real_world_suite import main
        main()

    def phase2() -> None:
        from rcwe.diagnostics.phase2_v07 import run_all_phase2_tracks
        run_all_phase2_tracks()

    def corpus() -> None:
        from rcwe.lm.corpus_trainer import main
        main()

    def custom() -> None:
        from rcwe.ingestion.custom_connector import main
        main()

    def v09() -> None:
        from rcwe.v09_runner import run_all_v09_tracks
        run_all_v09_tracks()

    return {
        "validate": ("multi-seed validation: plasticity, energy conservation, regime identification", validate),
        "extreme": ("context-length scaling sweep and symbolic conservation task", extreme),
        "dashboard": ("CSV/JSON ingestion, code-trace task and the static v0.5 dashboard", dashboard),
        "realworld": ("ETT / SWaT / NASA-style regime-shift suite", realworld),
        "phase2": ("Metal kernel scaling, self-healing controller, RCWE-Deep language model", phase2),
        "corpus": ("train RCWE-Deep on the repository's own source code", corpus),
        "custom": ("scan datasets/custom/ with the legacy predictor", custom),
        "v09": ("multimodal fusion, symplectic RL and wavefield swarm tracks", v09),
    }


def cmd_bench(args: argparse.Namespace) -> int:
    from rcwe.paths import output_dir

    suites = _bench_suites()
    if args.list or not args.suite:
        print("Benchmark suites (reports are written to ./rcwe_output or $RCWE_OUTPUT_DIR):\n")
        for name, (summary, _) in suites.items():
            print(f"  {name:<10} {summary}")
        print("\nRun one with `rcwe bench <suite>`, or all of them with `rcwe bench all`.")
        return EXIT_OK
    chosen = list(suites) if args.suite == "all" else [args.suite]
    for name in chosen:
        if name not in suites:
            print(f"error: unknown suite '{name}'. Choose from: {', '.join(suites)}, all", file=sys.stderr)
            return EXIT_ERROR
    for name in chosen:
        suites[name][1]()
        print()
    print(f"Reports are in {output_dir().resolve()}")
    return EXIT_OK


def cmd_demo_data(args: argparse.Namespace) -> int:
    from rcwe.toolkit.demo import write_demo_files

    paths = write_demo_files(_folder_target(args.out, "--out"), seed=args.seed)
    print("Wrote demo telemetry:")
    for key, path in paths.items():
        print(f"  {key:<9}{_show(path)}")
    print(f"\nTry: rcwe analyze {_quote(paths['incident'])} --train {_quote(paths['normal'])}")
    return EXIT_OK


def cmd_version(args: argparse.Namespace) -> int:
    import torch
    from rcwe import __version__

    print(f"rcwe {__version__} (PyTorch {torch.__version__})")
    return EXIT_OK


# --------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------
class _Parser(argparse.ArgumentParser):
    """A mistake on the command line exits with status 1: status 2 means "anomalies were found"."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="rcwe",
        description="Resonant Causal World Engine: find and explain anomalies in telemetry, and forecast it.",
        epilog="New here? Start with `rcwe quickstart`. Problems? Run `rcwe doctor`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, func: Callable[[argparse.Namespace], int], summary: str, **kwargs: Any) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=summary, description=summary, **kwargs)
        p.set_defaults(func=func)
        p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help="show full tracebacks on errors")
        return p

    p = add("quickstart", cmd_quickstart, "Guided tour on built-in demo data: train, detect, forecast, open the report.")
    p.add_argument("--out", default="rcwe_quickstart", help="folder for the demo files and results (default: ./rcwe_quickstart)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-open", action="store_true", help="do not open the report in a browser")
    p.add_argument("--quiet", action="store_true", help="no progress output")

    p = add("analyze", cmd_analyze, "One-shot: learn normal behaviour, scan the log, write report + model + scores.")
    p.add_argument("file", help="telemetry log (CSV, TSV, JSON, JSON-lines; .gz accepted)")
    p.add_argument("--train", metavar="FILE", help="separate log of known-good operation to learn from (recommended when you have one)")
    p.add_argument("--train-fraction", type=_number(float, 0.0, 1.0, open_ends=True), default=0.6, metavar="F",
                   help="without --train: learn from the first F of the rows (default 0.6)")
    p.add_argument("--train-rows", type=_number(int, 12), metavar="N", help="without --train: learn from the first N rows")
    p.add_argument("--out", metavar="DIR", help="results folder (default: ./rcwe_output/<file name>/)")
    p.add_argument("--title", help="report title")
    p.add_argument("--open", action="store_true", help="open the report in a browser when done")
    p.add_argument("--fail-on-anomaly", action="store_true", help="exit with status 2 if any event is found (for scripts and CI)")
    p.add_argument("--quiet", action="store_true", help="no progress output")
    _add_schema_options(p)
    _add_model_options(p)
    _add_detect_options(p)

    p = add("fit", cmd_fit, "Learn normal behaviour from a log and save a reusable model file.")
    p.add_argument("file", help="telemetry log of normal operation")
    p.add_argument("-o", "--output", metavar="MODEL", help="model file to write (default: <file name>.rcwe)")
    p.add_argument("--quiet", action="store_true", help="no progress output")
    _add_schema_options(p)
    _add_model_options(p)

    p = add("detect", cmd_detect, "Scan a log with a saved model; print explained events and optionally write a report.")
    p.add_argument("file", help="telemetry log to scan")
    p.add_argument("-m", "--model", required=True, help="model file from `rcwe fit` or `rcwe analyze`")
    p.add_argument("--report", metavar="HTML", help="write an interactive HTML report")
    p.add_argument("--json", metavar="FILE", help="write events and metrics as JSON")
    p.add_argument("--csv", metavar="FILE", help="write per-row scores and expected values as CSV")
    p.add_argument("--title", help="report title")
    p.add_argument("--open", action="store_true", help="open the report in a browser when done")
    p.add_argument("--fail-on-anomaly", action="store_true", help="exit with status 2 if any event is found")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--quiet", action="store_true", help="no progress output")
    _add_detect_options(p)

    p = add("forecast", cmd_forecast, "Forecast every sensor past the end of a log (or of the training log).")
    p.add_argument("file", nargs="?", help="recent telemetry to forecast from (default: the end of the training log)")
    p.add_argument("-m", "--model", required=True, help="model file")
    p.add_argument("--steps", type=_number(int, 1), default=48, help="how many steps ahead (default 48)")
    p.add_argument("--planned-controls", metavar="FILE", help="CSV with the planned control values for the forecast window")
    p.add_argument("--csv", metavar="FILE", help="write the forecast and its 80%% interval as CSV")
    p.add_argument("--json", metavar="FILE", help="write the forecast as JSON")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p = add("watch", cmd_watch, "Follow a growing CSV file and print an alert when an anomaly starts or clears.")
    p.add_argument("file", help="CSV file that another process appends rows to")
    p.add_argument("-m", "--model", required=True, help="model file")
    p.add_argument("--sensitivity", default="medium", metavar="LEVEL", help="low | medium | high, or a threshold in sigmas")
    p.add_argument("--adapt", action="store_true", help="keep learning from the stream (accept lasting changes as the new normal)")
    p.add_argument("--from-start", action="store_true", help="alert on rows already in the file (default: use them to warm up silently)")
    p.add_argument("--once", action="store_true", help="process what is in the file now, then exit")
    p.add_argument("--interval", type=_number(float, 0.0, open_ends=True), default=1.0, metavar="SEC",
                   help="how often to check for new rows (default 1s)")
    p.add_argument("--max-rows", type=_number(int, 1), metavar="N", help="stop after N rows")
    p.add_argument("--jsonl", metavar="FILE", help="append alerts to FILE as JSON lines (for other tools to pick up)")
    p.add_argument("--fail-on-anomaly", action="store_true", help="exit with status 2 if any alert was raised")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p = add("inspect", cmd_inspect, "Show how a file will be read: time axis, sensors, controls, gaps, ignored columns.")
    p.add_argument("file")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    _add_schema_options(p)

    p = add("serve", cmd_serve, "Start RCWE Studio: upload a log in the browser and explore the results.")
    p.add_argument("--host", default="127.0.0.1", help="address to bind (default: this machine only)")
    p.add_argument("--port", type=_number(int, 0, 65535), default=8765)
    p.add_argument("--no-open", action="store_true", help="do not open a browser window")

    p = add("ui", cmd_ui, "Start the Mission Control dashboard (live engine telemetry from the built-in sandbox).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=_number(int, 0, 65535), default=8766)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--export", metavar="HTML", help="write the standalone demo page instead of starting a server")

    p = add("bench", cmd_bench, "Run the research benchmark suites.")
    p.add_argument("suite", nargs="?", help="suite name, or 'all'")
    p.add_argument("--list", action="store_true", help="list the suites")

    p = add("demo-data", cmd_demo_data, "Write the built-in demo telemetry (normal + incident logs) as CSV files.")
    p.add_argument("--out", default="rcwe_demo", help="folder to write to (default: ./rcwe_demo)")
    p.add_argument("--seed", type=int, default=7)

    p = add("doctor", cmd_doctor, "Check that this machine can run RCWE and explain how to fix what cannot.")
    add("version", cmd_version, "Print the installed version.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:                       # --help (0) or a mistake on the command line (1)
        return int(stop.code or 0)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK
    debug = bool(getattr(args, "debug", False))
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - one place turns failures into a readable message
        if debug:
            raise
        from rcwe.toolkit import DataError, NotFittedError

        if isinstance(exc, (DataError, NotFittedError, FileNotFoundError, ValueError)):
            print(f"error: {exc}", file=sys.stderr)
        else:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("Run the command again with --debug for the full traceback, or `rcwe doctor` to check the installation.",
                  file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
