# RCWE User Guide

Everything you need to go from a telemetry file to explained anomalies and forecasts.

- [1. Install](#1-install)
- [2. The idea in one minute](#2-the-idea-in-one-minute)
- [3. Preparing your data](#3-preparing-your-data)
- [4. Command line](#4-command-line)
- [5. Python API](#5-python-api)
- [6. Reading the results](#6-reading-the-results)
- [7. Choosing settings](#7-choosing-settings)
- [8. Live data](#8-live-data)
- [9. RCWE Studio and Mission Control](#9-rcwe-studio-and-mission-control)
- [10. Using RCWE from scripts and CI](#10-using-rcwe-from-scripts-and-ci)
- [11. Troubleshooting](#11-troubleshooting)
- [12. Limitations](#12-limitations)

---

## 1. Install

RCWE needs Python 3.10 or newer and PyTorch 2.1 or newer. A GPU is optional; a laptop CPU handles
logs with millions of rows.

```bash
git clone https://github.com/stevegolter-Doradus-Labs/rcwe.git
cd rcwe
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Check the installation:

```bash
rcwe doctor
```

`doctor` verifies Python, PyTorch and NumPy, tries the engine on your accelerator (and tells you
if it will fall back to the CPU), checks that the output folder is writable, and trains a small
model end to end. Each problem comes with the command that fixes it.

Optional extras:

| Need | Install |
|---|---|
| Excel or Parquet files, pandas DataFrames | `pip install -e ".[data]"` |
| Running the test suite | `pip install -e ".[dev]"` |

To force a device, set `RCWE_DEVICE` (for example `RCWE_DEVICE=cpu`) or pass `--device cpu`.

---

## 2. The idea in one minute

1. **Learn normal.** `fit` trains the engine on a period of ordinary operation. It learns how each
   sensor moves from one sample to the next, given the other sensors and the control inputs.
2. **Compare.** `detect` replays a log. At every step the model says what it expects next; the
   difference from what actually arrived is scored in calibrated *sigmas*. A score of 4.5 means
   "this far from expectation happens by chance less than once in tens of thousands of readings".
3. **Explain.** Consecutive high scores are grouped into *events*. Each event names the sensors
   responsible, the direction of the deviation, and actual versus expected values.
4. **Look ahead.** `forecast` rolls the model forward, with intervals taken from backtests.

Try the whole loop on built-in data:

```bash
rcwe quickstart
```

---

## 3. Preparing your data

One row per time step, one column per signal. That is the only requirement.

| Accepted | Notes |
|---|---|
| `.csv`, `.tsv`, `.txt` | Comma, semicolon, tab or pipe separated; the delimiter is detected. |
| `.json` | A list of row objects, `{"telemetry": [...]}`, or an object of column lists. |
| `.jsonl`, `.ndjson` | One JSON object per line. |
| `.gz` versions of the above | Read directly. |
| `.xlsx`, `.parquet` | Needs the `data` extra. |
| In Python | Lists of dicts, dicts of columns, NumPy arrays, pandas DataFrames. |

How columns are interpreted:

| Column | Becomes | Detected by |
|---|---|---|
| Numeric measurement | **Sensor** | Any numeric column not claimed below. |
| Operator input, setpoint, actuator command | **Control** | Names starting `act_`, `u_`, `ctrl_`, `control_`, `cmd_`, `setpoint_`, `sp_` or ending `_setpoint`, `_cmd`, `_sp`, `_command`. |
| Timestamp | **Time axis** | Names like `timestamp`, `time`, `date`, `datetime`, `ts`; or any column of parseable dates; epoch seconds/milliseconds; or a step counter. |
| Ground truth | **Labels** (never used for training) | Names like `anomaly`, `label`, `fault`, `attack`, `regime_label`. Used only to score the result. |
| Text | Ignored | Listed in the notes. |

Always look before you train:

```bash
rcwe inspect your_log.csv
```

```text
your_log.csv
  rows        1,200
  time        timestamp   2026-03-12T10:00:00Z to 2026-03-16T13:55:00Z   every 300s
  sensors     6
    name                     min         mean          max  missing
    flow_m3h               32.86       58.266        91.94        0
    ...
  controls    act_valve_open, act_pump_speed
  labels      anomaly (237 rows marked anomalous)
```

If the guess is wrong, say so explicitly. Every command that reads a training file takes:

```bash
rcwe fit plant.csv --sensors "flow,pressure,temp" --controls "valve_position" --time-column logged_at
```

What RCWE does for you, and reports in the notes:

- Missing values and non-numeric cells are interpolated.
- Decimal commas and thousands separators (`1.013,2`) are understood.
- Rows out of chronological order are sorted.
- Explicit momentum columns from older RCWE logs (`p_0`, `p_1`, ...) are ignored; rates of change
  are derived from the sensors.
- **Running totals and random-walk signals** (energy counters, odometers) are recognised: only
  their rate of change is modelled, and they are not flagged merely for growing.
- **Constant columns** are kept: if one changes later, that is flagged.

**How much data?** A few hundred rows of normal operation is a reasonable minimum; a few thousand
is comfortable. Below 80 rows RCWE still runs, calibrates on the training rows themselves, and
warns that results are indicative. Fewer than 12 rows is refused.

---

## 4. Command line

Run `rcwe <command> --help` for the complete option list of any command. Add `--debug` to see a
full traceback instead of a one-line error.

### `rcwe analyze` — one file in, everything out

```bash
rcwe analyze your_log.csv --open
```

Learns normal behaviour from the first 60% of the log, scans all of it, and writes to
`./rcwe_output/your_log/`:

| File | Contents |
|---|---|
| `report.html` | The interactive report. |
| `result.json` | Events, sensor ranking, metrics. |
| `scores.csv` | One row per time step: score, event number, actual and expected value per sensor. |
| `model.rcwe` | The trained model, reusable with `detect`, `forecast` and `watch`. |
| `forecast.csv` | With `--forecast N`. |

Useful options:

| Option | Effect |
|---|---|
| `--train FILE` | Learn from a separate known-good log instead (recommended when you have one). |
| `--train-fraction 0.5`, `--train-rows 2000` | How much of the single file to learn from. |
| `--sensitivity low\|medium\|high\|<sigmas>` | Alarm threshold: 6.0, 4.5 (default), 3.5, or your own number. |
| `--adapt` | Adaptive mode (see [section 7](#7-choosing-settings)). |
| `--forecast 48` | Also forecast 48 steps past the end. |
| `--no-range-check` | Do not flag readings merely for being outside the training range. |
| `--out DIR`, `--title TEXT`, `--open` | Where to write, report title, open the report when done. |
| `--fail-on-anomaly` | Exit with status 2 if anything was found. |

### `rcwe fit` and `rcwe detect` — train once, reuse

```bash
rcwe fit normal_week.csv -o plant.rcwe
```

```bash
rcwe detect today.csv -m plant.rcwe --report today.html --csv today_scores.csv --json today.json
```

`detect` prints the events and writes only the outputs you ask for.

### `rcwe forecast`

```bash
rcwe forecast today.csv -m plant.rcwe --steps 48 --csv forecast.csv
```

Without a file, it forecasts from the end of the training log. If you know the planned control
values for the forecast window, pass them:

```bash
rcwe forecast today.csv -m plant.rcwe --steps 48 --planned-controls schedule.csv
```

### `rcwe watch` — follow a growing file

```bash
rcwe watch live.csv -m plant.rcwe --jsonl alerts.jsonl
```

Rows already in the file warm the model up silently; new rows are scored as they are appended.
An alert is printed when an anomaly starts and again when it clears. See
[section 8](#8-live-data).

### Other commands

| Command | Purpose |
|---|---|
| `rcwe quickstart` | Guided tour on demo data. |
| `rcwe inspect FILE` | Show how a file will be read (`--json` for machine-readable output). |
| `rcwe demo-data` | Write the demo CSV files to `./rcwe_demo/`. |
| `rcwe serve` | RCWE Studio in the browser. |
| `rcwe ui` | Mission Control dashboard. |
| `rcwe bench --list` | Research benchmark suites. |
| `rcwe doctor` | Installation check. |
| `rcwe version` | Installed version. |

The earlier `rcwe-bench` and `rcwe-predict` commands still work.

### Where files go

Anything you do not give an explicit path for is written under `./rcwe_output/`. Set
`RCWE_OUTPUT_DIR` to use another folder.

---

## 5. Python API

```python
from rcwe import TelemetryModel, load_table
```

### Train

```python
model = TelemetryModel().fit("normal_week.csv")
print(model.summary())
```

`fit` accepts a path, a list of row dicts, a dict of columns, a NumPy array `[rows, sensors]`, a
pandas DataFrame, or a `TelemetryTable`. Override column detection with `sensors=[...]`,
`controls=[...]`, `time_column="..."`.

Constructor options (all optional):

| Argument | Default | Meaning |
|---|---|---|
| `max_steps` | 400 | Upper bound on training steps; training stops early when validation stops improving. |
| `validation_fraction` | 0.2 | Share of the log, taken from the end, held out for early stopping and calibration. |
| `forecast_horizon` | 96 | Longest horizon backtested for forecast intervals. |
| `d_model` | `"auto"` | Width of the engine's latent space (64 up to 24 sensors, larger beyond). |
| `seed` | 0 | Makes training reproducible. |
| `device` | `"auto"` | `"cpu"`, `"cuda"`, `"mps"` or automatic. |

After training, `model.fit_summary_` holds the model card: rows used, skill against the naive
baseline overall and per sensor, forecast skill by horizon, warnings and notes.

### Detect

```python
result = model.detect("today.csv", sensitivity="medium")
print(result.summary())

for event in result.events:
    print(event.start_time, event.end_time, event.severity, event.kind, event.top_sensor)
    for sensor in event.sensors:
        print("   ", sensor["name"], sensor["direction"], sensor["actual"], "vs", sensor["expected"])
```

`result` fields you will use most:

| Field | Shape | Meaning |
|---|---|---|
| `events` | list | `AnomalyEvent` objects: `start`, `end`, `start_time`, `end_time`, `duration`, `peak_score`, `severity`, `kind`, `sensors`, `description`. |
| `scores` | `[rows]` | Overall anomaly score per step (the worst sensor). |
| `sensor_scores` | `[rows, sensors]` | Score per sensor. |
| `expected` | `[rows, sensors]` | What the model expected, in the sensor's own units. |
| `values` | `[rows, sensors]` | What actually arrived. |
| `is_anomaly` | `[rows]` | True inside an event. |
| `metrics` | dict or `None` | Present when the file has labels: incidents caught, false alarms, delay, precision, recall, AUROC. |

Exports: `result.to_html(path)`, `result.to_json(path)`, `result.to_csv(path)`, `result.to_dict()`.

Arguments of `detect`:

| Argument | Default | Meaning |
|---|---|---|
| `sensitivity` | `"medium"` | `"low"`, `"medium"`, `"high"`, or a threshold in sigmas. |
| `adapt` | `False` | Online-adaptive mode. |
| `range_check` | on in reference mode | Flag readings well outside the range seen in training. |
| `merge_gap` | 3 | Exceedances separated by at most this many quiet steps form one event. |
| `min_duration` | 1 | Minimum number of exceedances for an event. |

### Forecast

```python
forecast = model.forecast(steps=48, data="today.csv")
print(forecast.summary())
forecast.values        # [steps, sensors]
forecast.lower, forecast.upper          # 80% interval
forecast.lower95, forecast.upper95      # 95% interval
forecast.timestamps    # real timestamps when the log has a time axis
forecast.model_weight  # per sensor: 1 = pure model, 0 = fell back to "holds its latest value"
forecast.to_csv("forecast.csv")
```

Where the backtest made during `fit` shows the model cannot beat "the sensor holds its latest
value (plus its usual trend)", the forecast for that sensor is exactly that, and `model_weight`
says so. Controls are assumed to stay at their latest values unless you pass `future_controls`
(an array `[steps, controls]`, or a file or table with the control columns).

To show a forecast in the report, attach it first: `result.forecast = forecast`.

### Save and load

```python
model.save("plant.rcwe")
model = TelemetryModel.load("plant.rcwe")
```

One file holds the weights, scalers and calibration. It is read with PyTorch's `weights_only`
loader, so opening a model file from someone else cannot run code.

### Reading files yourself

```python
table = load_table("your_log.csv")
table.sensors, table.controls, table.time_column
table.values            # [rows, sensors], gaps filled
table.notes             # what was decided on the way in
train, rest = table.split(0.6)
```

---

## 6. Reading the results

### The summary

```text
#2 [CRITICAL] 2026-03-13T20:00:00Z to 2026-03-13T23:55:00Z (48 steps): pressure_bar ran below
   expectations (6.5 vs 7.41 expected at the peak), peaking at 23.6 sigma; also involved: vibration_mm_s.
```

- **Severity** compares the peak score with the threshold: low (under 1.5x), medium (under 2.5x),
  high (under 5x), critical.
- **The first sensor named** is the one with the sharpest evidence, which is usually where the
  problem is. Sensors listed as *also involved* disagree with it; they are often knock-on effects.
- **Kind** is one of:

| Kind | Meaning |
|---|---|
| spike | A jump lasting a few samples. |
| sustained deviation | The sensor stayed away from where the model expected it. |
| out of range | The reading left the envelope seen in training by more than a quarter of its width. The model is extrapolating there. |
| flatline | The reading stopped changing for much longer than it ever did in training. |

### The report

- **Anomaly score over time** with shaded events and, if your file has labels, a strip showing
  them. Hover for values; use the *Show* buttons or click an event row to zoom every chart.
- **Which sensors were involved**: steps each sensor spent above the threshold.
- **Readings against what the model expected**: actual (blue), expected (grey), the normal band,
  and the spans where that sensor was flagged.
- **Forecast**: recent history, forecast and its 80% interval.
- **How much to trust this**: skill per sensor against the naive baseline. A sensor at 0% is not
  being predicted any better than "it stays where it is"; its anomalies are still caught, but its
  forecast is just its last value.

### When the file has labels

If a column marks known anomalies, the summary and report compare detections with it: incidents
caught, false alarms, median detection delay, point-wise precision/recall/F1 and AUROC. Labels are
never used for training.

---

## 7. Choosing settings

### Sensitivity

| Level | Threshold | Use when |
|---|---|---|
| low | 6.0 sigma | Only unmistakable problems should raise an event. |
| medium | 4.5 sigma | The default. |
| high | 3.5 sigma | Missing a subtle fault is worse than investigating an extra alert. |

Any number of sigmas also works (`--sensitivity 5`).

### Reference or adaptive

| | Reference (default) | Adaptive (`--adapt`) |
|---|---|---|
| Compares each reading with | Behaviour learned during `fit`. | Behaviour learned so far, including during the scan. |
| A lasting change is flagged | For as long as it lasts. | When it starts (and when it ends), then accepted as the new normal. |
| Good for | Audits: "which periods were abnormal?" | Operations: "tell me when something changes, then stop nagging." |
| Speed | One vectorised pass. | Step by step (a few milliseconds per row). |

Adaptive mode is the engine's online plasticity at work: fast weights are updated from each
prediction error without backpropagation. The update happens after the prediction is scored, so
the model never sees an answer before being graded on it.

### Train on a clean period

The single biggest lever on quality. If the training rows contain an incident, the model partly
learns it as normal; you will still see where it starts and stops, but not its full extent.
Prefer `--train known_good.csv` whenever you have such a file.

---

## 8. Live data

### From a file that is being appended to

```bash
rcwe watch live.csv -m plant.rcwe --jsonl alerts.jsonl
```

```text
Watching live.csv with plant.rcwe (threshold 4.5 sigma, reference mode). Press Ctrl+C to stop.
  primed on 3,000 existing row(s); waiting for new ones ...
  ANOMALY   2026-03-13T20:00:00Z  score 23.6  pressure_bar: sudden jump below expected (6.5 vs 7.41)
  cleared   2026-03-13T23:55:00Z  (lasted 48 row(s), peak 23.6 sigma)
```

| Option | Effect |
|---|---|
| `--from-start` | Alert on rows already in the file too. |
| `--once` | Process what is there and exit (handy from cron). |
| `--adapt` | Keep learning from the stream. |
| `--interval 0.5` | How often to check for new rows, in seconds (default 1). |
| `--max-rows N` | Stop after N rows, counting the ones used to warm up. |
| `--jsonl FILE` | Append each alert and clearance as a JSON line for other tools. |
| `--fail-on-anomaly` | Exit status 2 if any alert was raised. |

A single reading just over the threshold is not announced; an alert needs two in a row or one
clearly above (1.25x the threshold). An alert is cleared after four quiet rows.

The file must have a header row naming the sensors the model was trained on; extra columns are
ignored, and the delimiter (comma, semicolon, tab or pipe) is taken from the header. A row is
scored once its line is complete. If the file is truncated or replaced, as log rotation does,
reading restarts from the top of the new file.

### From your own code

```python
monitor = model.stream(sensitivity="medium", adapt=False)

for row in incoming_rows:                 # dict: {column: value}
    step = monitor.update(row)
    if step.alert:
        worst = step.sensors[0]
        notify(f"{step.time}: {worst['name']} {worst['detail']} (score {step.score:.1f})")
```

`step.is_anomaly` is the raw per-reading flag; `step.alert` adds the debouncing described above.
`step.expected` and `step.values` hold the per-sensor numbers. Missing values in a row are carried
forward from the previous reading.

---

## 9. RCWE Studio and Mission Control

```bash
rcwe serve
```

Opens `http://127.0.0.1:8765`. Drop a log (and optionally a known-good log), choose the options,
and read the report in the page. Downloads for the report, events, scores, forecast and model are
offered when the analysis finishes. Files are posted to this local server only.

```bash
rcwe ui
```

Opens Mission Control: the engine running on its built-in sandbox next to a frozen Transformer.
The buttons mutate the simulation and every figure is measured on each tick. Dropping a telemetry
file on the page analyses it and links to the full report.

The server listens on `127.0.0.1` and has no login. It refuses requests that do not come from its
own pages. Use `--host` to expose it only on a network you trust.

---

## 10. Using RCWE from scripts and CI

Exit status:

| Status | Meaning |
|---|---|
| 0 | Success. |
| 1 | Error (bad input, missing file, ...); the message is on stderr. |
| 2 | `--fail-on-anomaly` was given and at least one event or alert was found. |

A nightly check:

```bash
rcwe detect /data/today.csv -m /models/plant.rcwe --json /reports/today.json --report /reports/today.html --quiet --fail-on-anomaly
```

`result.json` is stable to parse: `event_count`, `events[]` (with `start_time`, `end_time`,
`severity`, `kind`, `sensors[]`, `description`), `sensor_ranking[]`, `metrics`.

---

## 11. Troubleshooting

Start with `rcwe doctor`.

| Symptom | What to do |
|---|---|
| `No numeric sensor columns were found` | The delimiter or header was not recognised. Run `rcwe inspect`; pass `--sensors` explicitly. |
| `Sensor 'x' that this model was trained on is missing` | The new file must contain the sensors the model was trained on, with the same names. |
| Everything is flagged from the first row | The new log is on a different scale or operating point than the training log. Train on data from the same regime, or use `--adapt`. |
| One giant event hides later ones | Something stays wrong for the rest of the log (a changed constant, a permanent offset). Look at the per-sensor panels in the report, or scan the later part separately. |
| Many short low-severity events | Lower the sensitivity (`--sensitivity low`), or raise `--train-rows` so more normal variation is learned. |
| `out of range` events you do not care about | Add `--no-range-check`. |
| Skill is near 0% | The signals are close to unpredictable step to step, or the log is too short. Detection still works; do not rely on the forecast. |
| Training is slow or fails on a GPU | Pass `--device cpu`. For logs this size the CPU is usually as fast. |
| Report shows fewer points than rows | Long logs are thinned for display: the highest score of every stretch is kept, and the rows around the strongest events are kept in detail. `scores.csv` has every row. |

---

## 12. Limitations

- **One step of memory.** The model sees each sensor's current level and rate of change, plus the
  controls. It does not learn daily or weekly seasonality by itself; where that matters, include
  the driving signal (time of day, load, setpoints) as a column.
- **Evenly spaced rows are assumed.** Irregular sampling is noted, not corrected.
- **Unsupervised.** RCWE flags departures from learned behaviour. Whether a departure is a fault
  or a legitimate new operating mode is your call; adaptive mode exists for the second case.
- **Attribution is evidence, not proof.** The first-named sensor is where the deviation is
  sharpest. A fault that shows only through its effect on other sensors will be attributed to them.
- **Forecast intervals are empirical.** They reflect how the model did on the held-out part of the
  training log, and assume the future resembles it.
