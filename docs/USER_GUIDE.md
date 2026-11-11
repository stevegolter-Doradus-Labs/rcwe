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
| `.csv`, `.tsv`, `.txt` | Comma, semicolon, tab or pipe separated; the delimiter is detected. A header row is expected but not required. Lines starting with `#` are skipped. |
| `.json` | A list of row objects, `{"telemetry": [...]}`, or an object of column lists. |
| `.jsonl`, `.ndjson` | One JSON object per line (also recognised by content, whatever the file is called). |
| `.gz` versions of the above | Read directly. |
| `.xlsx`, `.parquet` | Needs the `data` extra. |
| In Python | Lists of dicts, dicts of columns, NumPy arrays, pandas DataFrames. |

How columns are interpreted:

| Column | Becomes | Detected by |
|---|---|---|
| Numeric measurement | **Sensor** | Any numeric column not claimed below. |
| Operator input, setpoint, actuator command | **Control** | Names starting `act_`, `u_`, `ctrl_`, `control_`, `cmd_`, `setpoint_`, `sp_` or ending `_setpoint`, `_cmd`, `_sp`, `_command`. |
| Timestamp | **Time axis** | Names like `timestamp`, `time`, `date`, `datetime`, `ts`; longer names that contain such a word (`timestamp_ms`, `Time [s]`) when the values are clock times; any column of parseable dates; a step counter. |
| Ground truth | **Labels** (never used for training) | Names like `anomaly`, `label`, `fault`, `attack`, `regime_label`. Used only to score the result. |
| Text | Ignored | Listed in the notes. |

Always look before you train:

```bash
rcwe inspect your_log.csv
```

```text
demo/pump_station_incident.csv
  rows        1,200
  time        timestamp   2026-03-12T10:00:00Z to 2026-03-16T13:55:00Z   every 300s
  sensors     6
    name                     min         mean          max  missing
    flow_m3h               32.86       58.266        91.94        0
    pressure_bar           6.132       7.0503        7.609        0
    ...
  controls    act_valve_open, act_pump_speed
  labels      anomaly (237 rows marked anomalous)
```

If the guess is wrong, say so explicitly. Every command that reads a training file takes:

```bash
rcwe fit plant.csv --sensors "flow,pressure,temp" --controls "valve_position" --time-column logged_at
```

A model remembers these choices: later `detect`, `forecast` and `watch` calls read the same
columns the same way, without the options being repeated.

### What RCWE works out for you

Each of these decisions is reported in the notes that `inspect`, `fit` and the report print.

**Time**

- Dates are read with **one format for the whole column**. `11/03/2026` cannot be told from its
  day/month twin on its own, so the reading that parses every cell, keeps the rows in order and
  gives the most even spacing is chosen, and the choice is stated ("dates read as
  day/month/year"). If both readings fit equally well, the note says so; ISO dates
  (`2026-03-11`) remove the doubt.
- Unix time is recognised in seconds, milliseconds, microseconds and nanoseconds, as are
  compact stamps such as `20260311105800`, and a date column next to a time-of-day column.
- Rows out of chronological order are sorted. The hour that local clocks repeat when daylight
  saving ends is kept in file order instead of being shuffled together.
- A timestamp that cannot be read is interpolated from its neighbours.

**Numbers**

- Decimal commas are understood, and told apart from thousands separators by looking at the
  whole column: in `994, 998, "1,002", "1,006"` the comma is a thousands separator; next to
  `0,998` it is a decimal mark. When a column leaves real doubt, the note says how it was read.
- Percent signs, non-breaking spaces and the usual missing-value markers (`NA`, `null`, `-`, ...)
  are handled.

**Sensors**

- **Gaps.** For training, missing values are interpolated. When a log is scanned, a missing
  reading is not scored and the last real value is held in its place; the first reading after a
  gap is judged against how far that sensor can move in that many steps. An interpolated value
  would already know what comes after the gap.
- **Running totals and random-walk signals** (energy counters, odometers) are recognised: only
  their rate of change is modelled, and they are not flagged merely for growing. A counter that
  moves by exactly the same amount every row is flagged the moment it skips.
- **Constant columns** are kept: if one changes later, that is flagged, whatever its units.
- **Status and mode flags** (a handful of distinct values) switch without warning; a switch that
  the training log shows at least three times is treated as normal.
- **Wild readings in the training log** (a `1e9` glitch) are set aside when each sensor's scale
  is learned, so one of them cannot blind a sensor.
- Explicit momentum columns from older RCWE logs (`p_0`, `p_1`, ... next to `timestamp_step`)
  are ignored; rates of change are derived from the sensors.

**Labels.** Numeric labels count as anomalous when non-zero (`-1`/`1` is read as outlier/inlier).
Text labels use the usual words (`normal`, `ok`, `good`, ...); where none is recognised, the most
frequent value is taken as normal and the note says which. Extra label-like columns are left out
rather than fed to the model.

### Controls: reacting or commanding?

A control column can be one of two things, and only you know which.

- A **command** is set independently of the sensors: a setpoint, a schedule, an operator's
  input. The readings respond to it.
- A **controller output** reacts to the sensors, often within the same sample.

By default RCWE assumes the cautious case. It gives the model a control's value only up to the
row it predicts *from*, never for the row being predicted, because a controller output would give
the answer away: the model would learn "the valve command jumps when the pressure drops" and
explain a leak as expected. The price is that the model cannot anticipate what a control change
does, so forecasts ignore a planned change, and a large setpoint step can itself raise an event.

Tell RCWE which controls are commands and it uses them in full:

```bash
rcwe fit normal_week.csv --commands "valve_setpoint,pump_speed" -o plant.rcwe
```

`--commands all` declares every control; `--commands none` declares none. Controls whose names
say so (`setpoint_*`, `sp_*`, `*_setpoint`, `*_sp`) are commands without being listed.

What changes, measured on the demo plant with an operator stepping the valve and pump speed by
hand ([BENCHMARKS.md, section 2](BENCHMARKS.md)): a planned +0.10 valve step moved the 24-step
flow forecast by 0% of the plant's true response by default and by 87% with the controls declared
as commands, and one-step skill rose from about 25% to 33%. The model learns a response only
from changes of that kind in the training log.

Declaring commands does not change the forecasts you ask for without a plan. A model that is told
its commands learns to lean on them, and with no plan all a forecast could tell it is that they
stay where they are, which is a poor guess for a schedule that keeps moving. So when commands are
declared, `fit` trains a second, companion model that is never told them, and a forecast made
without `--planned-controls` comes from that one (its notes say so). The price is time: training
runs twice.

Either way, an event that begins as a control changes markedly says so, which is usually enough
to tell an operator's action from a fault:

```text
motor_power_kw jumped above expectations (29.29 vs 26.33 expected), peaking at 5.8 sigma;
coincides with a change in act_pump_speed (0.6 to 0.65).
```

### How much data?

A few hundred rows of normal operation is a reasonable minimum; a few thousand is comfortable.
Below 80 rows RCWE still runs, calibrates on the training rows themselves, and warns that results
are indicative. Fewer than 12 rows is refused. Files with more than 10,000 columns are refused.

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
| `--commands A,B\|all\|none` | Which controls are commands (see [section 3](#controls-reacting-or-commanding)). |
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
values for the forecast window, pass them; a file holding just the control columns is enough:

```bash
rcwe forecast today.csv -m plant.rcwe --steps 48 --planned-controls schedule.csv
```

A plan moves the forecast through the controls that were declared as commands when the model
was trained; the forecast's notes say which controls it could not take into account.

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
`controls=[...]`, `time_column="..."`, `label_column="..."`, and declare commands with
`commands=[...]` (or `True` / `"all"`, `False` / `"none"`).

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
baseline overall and per sensor, forecast skill by horizon, which controls are commands,
warnings and notes.

### Detect

```python
result = model.detect("today.csv", sensitivity="medium")
print(result.summary())

for event in result.events:
    print(event.start_time, event.end_time, event.severity, event.kind, event.top_sensor)
    for sensor in event.sensors:
        print("   ", sensor["name"], sensor["direction"], sensor["actual"], "vs", sensor["expected"])
    for change in event.control_changes:
        print("    control", change["name"], "went from", change["before"], "to", change["after"])
```

The data is read with the columns chosen for `fit`, wherever it has them.

`result` fields you will use most:

| Field | Shape | Meaning |
|---|---|---|
| `events` | list | `AnomalyEvent` objects: `start`, `end`, `start_time`, `end_time`, `duration`, `peak_score`, `severity`, `kind`, `sensors`, `control_changes`, `description`. |
| `scores` | `[rows]` | Overall anomaly score per step (the worst sensor). |
| `sensor_scores` | `[rows, sensors]` | Score per sensor. |
| `expected` | `[rows, sensors]` | What the model expected, in the sensor's own units. |
| `values` | `[rows, sensors]` | The readings that were scored (a gap shows the value held across it). |
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
forecast.timestamps    # continues the log's own time axis
forecast.model_weight  # per sensor: 1 = pure model, 0 = fell back to "holds its latest value"
forecast.to_csv("forecast.csv")
```

Where the backtest made during `fit` shows the model cannot beat "the sensor holds its latest
value (plus its usual trend)", the forecast for that sensor is exactly that, and `model_weight`
says so. Controls are assumed to stay at their latest values unless you pass `future_controls`
(an array `[steps, controls]`, or a file, table or list of rows with the control columns). A model
with commands makes a forecast without a plan through its companion model
([section 3](#controls-reacting-or-commanding)).

To show a forecast in the report, attach it first: `result.forecast = forecast`.

### Save and load

```python
model.save("plant.rcwe")
model = TelemetryModel.load("plant.rcwe")
```

One file holds the weights, scalers and calibration. It is read with PyTorch's `weights_only`
loader, so opening a model file from someone else cannot run code, and it carries a checksum, so
a damaged file is refused instead of quietly giving different scores.

### Reading files yourself

```python
table = load_table("your_log.csv")
table.sensors, table.controls, table.time_column
table.values            # [rows, sensors], gaps filled
table.missing           # [rows, sensors], True where a value was absent
table.notes             # what was decided on the way in
train, rest = table.split(0.6)
```

---

## 6. Reading the results

### The summary

```text
#2 [CRITICAL] 2026-03-13T20:00:00Z to 2026-03-13T23:50:00Z (47 steps): pressure_bar ran below
   expectations (6.5 vs 7.42 expected at the peak), peaking at 24.2 sigma; also involved: vibration_mm_s.
```

- **Severity** compares the peak score with the threshold: low (under 1.5x), medium (under 2.5x),
  high (under 5x), critical.
- **The first sensor named** is the one with the sharpest evidence, which is usually where the
  problem is. Sensors listed as *also involved* deviated strongly as well; they are often
  knock-on effects.
- **"coincides with a change in ..."** is added when a control input changed by much more than
  it usually does as the event began or within the three rows before it.
- **Kind** is one of:

| Kind | Meaning |
|---|---|
| spike | A jump lasting a few samples. |
| sustained deviation | The sensor stayed away from where the model expected it. |
| out of range | The reading left the envelope seen in training by more than a quarter of its width. The model is extrapolating there. The value given is the most extreme one reached. |
| flatline | The reading stopped changing for much longer than it ever did in training. |

The closing line `Most time above threshold: ...` lists where deviations lasted longest over the
whole log. It can differ from the sensors the events name first, which are where they started.

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

The report is one file with no outside dependencies. It shows the analysed file's name, not the
folder it was in.

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
Prefer `--train known_good.csv` whenever you have such a file. Isolated glitches are a different
matter: a few wild readings in the training log do not distort what the model learns.

### Declare your commands

If your control columns are setpoints or schedules, say so with `--commands`
([section 3](#controls-reacting-or-commanding)). It is what lets forecasts respond to a plan.
Training then takes about twice as long, because a companion model is trained for the forecasts
you ask for without a plan.

---

## 8. Live data

### From a file that is being appended to

```bash
rcwe watch live.csv -m plant.rcwe --jsonl alerts.jsonl
```

```text
Watching live.csv with plant.rcwe (threshold 4.5 sigma, reference mode). Press Ctrl+C to stop.
  primed on 3,000 existing row(s); waiting for new ones ...
  ANOMALY   2026-03-13T20:00:00Z  score 24.2  pressure_bar: sudden jump below expected (6.5 vs 7.42)
  cleared   2026-03-13T23:50:00Z  (lasted 46 row(s), peak 24.2 sigma)
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
clearly above (1.25x the threshold). An alert is cleared after four quiet rows. If a control
changed markedly in the row that raised the alert, the line says so.

The file must have a header row naming the sensors the model was trained on; extra columns are
ignored, lines starting with `#` are skipped, and the delimiter (comma, semicolon, tab or pipe) is
taken from the header. A row is scored once its line is complete. If the file is truncated or
replaced, as log rotation does, reading carries on from the top of the new content with the same
columns.

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
`step.expected` and `step.values` hold the per-sensor numbers and `step.controls_changed` names
any control that moved markedly in this row. A value that is missing or not a finite number is
"no reading": the last real value is held and that sensor is not scored for the row. A monitor
gives the same scores as `detect` on the same rows.

---

## 9. RCWE Studio and Mission Control

```bash
rcwe serve
```

Opens `http://127.0.0.1:8765`. Drop a log (and optionally a known-good log), choose the options,
and read the report in the page. Downloads for the report, events, scores, forecast and model are
offered when the analysis finishes. Files are posted to this local server only. The two files
together may be up to 56 MB; use `rcwe analyze` for larger ones. Studio keeps the 25 most recent
analyses in the output folder and removes older ones, so download what you want to keep.

```bash
rcwe ui
```

Opens Mission Control: the engine running on its built-in sandbox next to a frozen Transformer.
The buttons mutate the simulation and every figure is measured on each tick. Dropping a telemetry
file on the page analyses it and links to the full report.

The server listens on `127.0.0.1` and has no login. It answers its API only to its own pages,
checks the address it was reached by, and does not let other sites frame those pages. Use
`--host` to expose it only on a network you trust.

---

## 10. Using RCWE from scripts and CI

Exit status:

| Status | Meaning |
|---|---|
| 0 | Success. |
| 1 | Error: bad input, a missing file, or a mistake on the command line. The message is on stderr. |
| 2 | `--fail-on-anomaly` was given and at least one event or alert was found. |

A nightly check:

```bash
rcwe detect /data/today.csv -m /models/plant.rcwe --json /reports/today.json --report /reports/today.html --quiet --fail-on-anomaly
```

`result.json` is stable to parse: `event_count`, `events[]` (with `start_time`, `end_time`,
`severity`, `kind`, `sensors[]`, `control_changes[]`, `description`), `sensor_ranking[]`,
`metrics`. Readings are written with twelve significant digits; values that are not finite are
`null`.

---

## 11. Troubleshooting

Start with `rcwe doctor`.

| Symptom | What to do |
|---|---|
| `No numeric sensor columns were found` | The delimiter or header was not recognised. Run `rcwe inspect`; pass `--sensors` explicitly. |
| `Sensor 'x' that this model was trained on is missing` | The new file must contain the sensors the model was trained on, with the same names. |
| Dates come out in the wrong order, or as the wrong month | Check the note that says how they were read. Dates like `03/04/2026` fit two readings; ISO dates (`2026-04-03`) fit one. |
| A number is 1000 times too large or too small | Check the notes for how a thousands separator or decimal mark was read; write numbers without thousands separators to be sure. |
| Everything is flagged from the first row | The new log is on a different scale or operating point than the training log. Train on data from the same regime, or use `--adapt`. |
| One giant event hides later ones | Something stays wrong for the rest of the log (a changed constant, a permanent offset). Look at the per-sensor panels in the report, or scan the later part separately. |
| Many short low-severity events | Lower the sensitivity (`--sensitivity low`), or raise `--train-rows` so more normal variation is learned. |
| An event every time a setpoint is changed | The event says which control moved. Declare that control with `--commands` and train on a log that contains such changes. |
| A planned control change does not move the forecast | The control was not declared as a command when the model was trained (the forecast's notes name it). |
| Training takes twice as long as it did | Commands were declared, or picked up from names such as `*_setpoint`, so a companion model is trained as well. `--commands none` turns both off. |
| `out of range` events you do not care about | Add `--no-range-check`. |
| Skill is near 0% | The signals are close to unpredictable step to step, or the log is too short. Detection still works; do not rely on the forecast. |
| `... was written by an earlier build of RCWE` | Model files from before this release cannot be read; train the model again with `rcwe fit`. |
| `... is damaged: its contents do not match the checksum` | The model file was altered or truncated after it was saved. Copy it again or retrain. |
| `RCWE_DEVICE asks for CUDA, but ...` | The device you asked for is not available here. Unset `RCWE_DEVICE`, or pass `--device cpu`. |
| Training is slow or fails on a GPU | Pass `--device cpu`. For logs this size the CPU is usually as fast. |
| Report shows fewer points than rows | Long logs are thinned for display: the highest score of every stretch is kept, every event keeps its outline, and the strongest events are kept in detail. `scores.csv` has every row. |

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
- **Controls are only as useful as you make them.** Unless a control is declared as a command,
  the model does not anticipate what a change in it will do. Declared, it learns the response
  from the changes the training log contains and reproduces most of it, not all.
- **The first two rows of a scan cannot surprise.** There is nothing to compare the first with,
  and the second is predicted from a guessed rate of change. Range and stuck-sensor checks apply
  from the first row.
- **Forecast intervals are empirical.** They reflect how the model did on the held-out part of the
  training log with controls held where they were, and assume the future resembles it.
