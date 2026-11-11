"""
`TelemetryModel` -- the five-line way to use the Resonant Causal World Engine on your data.

    from rcwe import TelemetryModel

    model = TelemetryModel().fit("normal_operation.csv")     # learn what "normal" looks like
    result = model.detect("today.csv")                        # find and explain anomalies
    print(result.summary()); result.to_html("report.html")
    forecast = model.forecast(steps=48)                       # look ahead, with intervals
    model.save("plant.rcwe")                                  # reuse later: TelemetryModel.load(...)

How it works
------------
Each sensor reading q_t is paired with its velocity p_t (the change since the previous
sample), giving the phase-space state (q, p) the engine was designed around. The engine is
trained to predict the next state from the current state and the control inputs. Four
things keep the results honest:

* No target leakage. Predictions are always made *before* the true next value is shown to
  the engine (the engine's posterior world-line weighting is only used for the optional
  online-adaptation update, after scoring). A reading that was missing is never scored, and
  the reading after a gap is compared with the last real one, not with an interpolation
  that already knows it.
* Controls are only trusted as far as they can be. A control column may itself react to the
  sensors (a controller output), so by default the model sees control values only up to the
  row it predicts *from*. Columns declared as commands (setpoints, schedules: values chosen
  independently of the sensors) are also given for the row being predicted. A forecast that
  comes with no plan for the commands is made by a companion model that is never told them,
  so declaring commands cannot make such a forecast worse.
* A naive baseline is always in the room. Accuracy is reported as skill versus "next value
  = current value", and a per-sensor calibration shrinks the engine's prediction toward
  that baseline wherever the engine does not beat it on held-out data.
* Calibration happens on data the engine was not trained on (the last part of the log), with
  statistics that a handful of wild readings cannot move.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.kernels.accelerator import describe_accelerator, select_best_device
from rcwe.toolkit.results import AnomalyEvent, DetectionResult, FitSummary, Forecast, _fmt, _fmt_pair
from rcwe.toolkit.scoring import (
    REASON_TEXT, REASONS, ScoreCalibration, detection_metrics, ewma, find_events,
    longest_run, resolve_threshold, robust_sigma, score_residuals,
)
from rcwe.toolkit.table import DataError, TelemetryTable, format_time, load_like, load_table

__all__ = ["TelemetryModel", "StreamMonitor", "StepResult", "NotFittedError"]

FORMAT_NAME = "rcwe-telemetry-model"
FORMAT_VERSION = 2
_CLIP = 8.0            # model inputs are clipped to +/- this many (normalised) units
_Z_CAP = 1.0e4         # scores are capped here so that broken sensors do not print as 1e17
_CHUNK = 8192          # rows per forward pass when scoring long logs
_CONTEXT = 128         # rows of history used to warm the standing wavefield before forecasting
_TARGET_CLIP = 25.0    # calibration ignores how far beyond this a (standardised) change went
_EWMA_LAMBDA = 0.25
ProgressFn = Callable[[str, float, str], None]


class NotFittedError(RuntimeError):
    """Raised when a model is used before `fit()` or `load()`."""


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _body_bounds(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per column, the limits beyond which a value is wild: more than 1.5 times the width of the
    range that holds 99% of the column away from that range. Statistics are taken after
    clipping to these limits, so that a 1e9 glitch cannot set a sensor's scale, while rare but
    real operating states (anything inside or near the body) count in full.
    """
    x = np.asarray(x, dtype=float)
    cols = x.shape[1]
    if x.shape[0] < 8:
        return np.full(cols, -np.inf), np.full(cols, np.inf)
    lo = np.quantile(x, 0.005, axis=0, method="higher")
    hi = np.quantile(x, 0.995, axis=0, method="lower")
    slack = np.where(hi > lo, 1.5 * (hi - lo), np.inf)       # a column that barely varies cannot be judged
    return lo - slack, hi + slack


def _clipped(x: np.ndarray) -> np.ndarray:
    lo, hi = _body_bounds(x)
    return np.clip(x, lo, hi)


def _envelope(qn: np.ndarray, n_train: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per sensor: (does it keep to a stable operating envelope, has that envelope settled).

    Not stable -- level hidden from the engine, range check off -- are signals with a strong
    trend (counters, totalisers) and signals whose held-out tail already left the envelope of
    the training part. Not settled are signals whose range keeps growing with the amount of
    data (each quarter of the training log spans much less than the whole). That is true of
    a random walk, but just as true of an ordinary daily cycle of which the log holds only a
    period or two, so it is no ground for switching anything off: the range check merely
    allows such a sensor more headroom.
    Logs that are too short to judge are given the benefit of the doubt.
    """
    T, n = qn.shape
    level_ok = np.ones(n, dtype=bool)
    range_ok = np.ones(n, dtype=bool)
    if n_train < 80:
        return level_ok, range_ok
    train = qn[:n_train]
    width = np.ptp(train, axis=0)
    steps = np.arange(n_train, dtype=float)
    for i in range(n):
        if width[i] <= 0:
            continue
        with np.errstate(invalid="ignore"):
            trend = abs(float(np.corrcoef(steps, train[:, i])[0, 1]))
        blocks = np.array([np.ptp(b) for b in np.array_split(train[:, i], 4)])
        if np.isfinite(trend) and trend >= 0.9:
            level_ok[i] = False
            continue
        if width[i] > 1.4 * float(np.median(blocks)):
            range_ok[i] = False
        if n_train < T:
            margin = max(0.25 * width[i], 0.25)
            tail = qn[n_train:, i]
            if tail.max() > train[:, i].max() + margin or tail.min() < train[:, i].min() - margin:
                level_ok[i] = False
    return level_ok, range_ok


def _hold_gaps(values: np.ndarray, missing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For scoring: put the last real reading in every cell that was missing (the first real
    reading, for a gap at the very start), and count for every cell how many missing cells
    came just before it. A gap is then never filled with knowledge of what follows it.
    """
    held = np.array(values, dtype=float, copy=True)
    gap = np.zeros(values.shape)
    T = values.shape[0]
    rows = np.arange(T)
    for i in np.flatnonzero(missing.any(axis=0)):
        real = ~missing[:, i]
        if not real.any():
            continue
        last = np.maximum.accumulate(np.where(real, rows, -1))          # last real reading at or before each row
        source = np.where(last >= 0, last, int(np.argmax(real)))
        held[:, i] = values[source, i]
        before = np.concatenate([[-1], last[:-1]])                      # last real reading strictly before each row
        gap[:, i] = np.where(real & (before >= 0), rows - before - 1, 0)
    return held, gap


def _digest(payload: Dict[str, Any]) -> str:
    """Fingerprint of every number in a model file, so that a damaged file is refused rather than trusted."""
    sha = hashlib.sha256()

    def feed(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            sha.update(str(tuple(value.shape)).encode())
            sha.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                sha.update(str(key).encode())
                feed(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                feed(item)
        else:
            sha.update(repr(value).encode())

    feed({key: value for key, value in payload.items() if key != "digest"})
    return sha.hexdigest()


_CONTROLS_HELD = "Controls were assumed to stay at their latest values."


def _is_setpoint_name(name: str) -> bool:
    low = name.lower()
    return low.startswith(("setpoint_", "sp_")) or low.endswith(("_setpoint", "_sp"))


def _severity(ratio: float) -> str:
    if ratio < 1.5:
        return "low"
    if ratio < 2.5:
        return "medium"
    if ratio < 5.0:
        return "high"
    return "critical"


class StepResult:
    """Outcome of feeding one new row to a `StreamMonitor`."""

    __slots__ = ("index", "time", "score", "is_anomaly", "alert", "sensors", "expected", "values", "sensor_scores",
                 "controls_changed")

    def __init__(self, index: int, time: str, score: float, is_anomaly: bool, sensors: List[Dict[str, Any]],
                 expected: Dict[str, float], values: Dict[str, float], sensor_scores: Dict[str, float], alert: bool = False,
                 controls_changed: Optional[List[str]] = None):
        self.index = index
        self.time = time
        self.score = score
        self.is_anomaly = is_anomaly      # this reading is above the threshold
        self.alert = alert                # ...and it is not a one-off blip (two in a row, or clearly above)
        self.sensors = sensors            # sensors above threshold, worst first: {name, score, reason, direction, detail}
        self.expected = expected
        self.values = values
        self.sensor_scores = sensor_scores
        self.controls_changed = controls_changed or []     # control inputs that changed markedly in this row

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "time": self.time, "score": round(self.score, 4), "is_anomaly": self.is_anomaly,
            "alert": self.alert, "sensors": self.sensors, "expected": self.expected, "values": self.values,
            "sensor_scores": {k: round(v, 4) for k, v in self.sensor_scores.items()},
            "controls_changed": list(self.controls_changed),
        }

    def __repr__(self) -> str:
        flag = "ANOMALY" if self.is_anomaly else "ok"
        worst = f" ({self.sensors[0]['name']}: {self.sensors[0]['reason']})" if self.sensors else ""
        return f"<StepResult #{self.index} {self.time} score={self.score:.2f} {flag}{worst}>"


class TelemetryModel:
    """
    Learn normal behaviour from telemetry, then detect anomalies and forecast.

    Parameters
    ----------
    d_model : int or "auto"
        Width of the engine's complex latent space. "auto" picks 64 for up to 24 sensors and
        grows with the number of sensors after that.
    max_steps : int
        Upper bound on training steps. Training stops early once validation stops improving.
    window, batch_size, learning_rate, weight_decay
        Optimisation settings; the defaults work well from a few hundred to millions of rows.
    validation_fraction : float
        Share of the log (taken from the end) held out for early stopping and calibration.
    forecast_horizon : int
        Longest horizon that is backtested during `fit` to calibrate forecast intervals.
    seed : int
        Makes training reproducible.
    device : "auto" | "cpu" | "cuda" | "mps"
    """

    def __init__(
        self,
        d_model: Union[int, str] = "auto",
        num_worldlines: int = 3,
        num_freq_bands: int = 16,
        max_steps: int = 400,
        window: int = 32,
        batch_size: int = 32,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-2,
        validation_fraction: float = 0.2,
        patience: int = 150,
        forecast_horizon: int = 96,
        seed: int = 0,
        device: str = "auto",
    ):
        if isinstance(d_model, str) and d_model != "auto":
            raise ValueError("d_model must be an integer or 'auto'.")
        if not 0.0 <= validation_fraction < 0.9:
            raise ValueError("validation_fraction must be in [0, 0.9).")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be at least 1.")
        self.settings: Dict[str, Any] = {
            "d_model": d_model, "num_worldlines": int(num_worldlines), "num_freq_bands": int(num_freq_bands),
            "max_steps": int(max_steps), "window": int(window), "batch_size": int(batch_size),
            "learning_rate": float(learning_rate), "weight_decay": float(weight_decay),
            "validation_fraction": float(validation_fraction), "patience": int(patience),
            "forecast_horizon": int(forecast_horizon), "seed": int(seed), "device": str(device),
        }
        self.engine: Optional[ResonantCausalWorldEngine] = None
        self.sensors: List[str] = []
        self.controls: List[str] = []
        self.commands: List[str] = []                 # controls whose value is known for the row being predicted
        self.time_column: Optional[str] = None
        self.label_column: Optional[str] = None
        self.fit_summary_: Optional[FitSummary] = None
        self._device = torch.device("cpu")
        self._mean = self._std = self._vs = self._vmean = self._cmean = self._cstd = self._gain = None
        self._cds: Optional[np.ndarray] = None          # usual size of a control's step change (normalised units)
        self._cnote: Optional[np.ndarray] = None        # a control change larger than this is worth pointing out
        self._shift: Optional[np.ndarray] = None        # 1 for commands (use the next row's value), 0 otherwise
        self._level_mask: Optional[np.ndarray] = None   # 0 hides a non-stationary sensor's level from the engine
        self._active: Optional[np.ndarray] = None       # 0 for sensors that never varied during training
        self._cal: Optional[ScoreCalibration] = None
        self._fc: Optional[Dict[str, np.ndarray]] = None
        self._tail: Optional[TelemetryTable] = None     # the end of the training log, for forecasts that continue it
        self._tail_offset = 0
        self._fit_table: Optional[TelemetryTable] = None
        self._free: Optional["TelemetryModel"] = None   # trained without commands; forecasts for which no plan is given

    # ------------------------------------------------------------------------------
    # basics
    # ------------------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self.engine is not None and self._cal is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError("This model has not been trained yet. Call fit(...) or TelemetryModel.load(...) first.")

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "TelemetryModel(unfitted)"
        skill = self.fit_summary_.skill if self.fit_summary_ else float("nan")
        return (f"TelemetryModel({len(self.sensors)} sensors, {len(self.controls)} controls, "
                f"skill vs persistence {skill * 100:+.0f}%)")

    def summary(self) -> str:
        self._require_fitted()
        return self.fit_summary_.summary() if self.fit_summary_ else repr(self)

    def _resolve_device(self) -> torch.device:
        choice = self.settings["device"]
        if choice in ("auto", None, ""):
            return select_best_device()
        return select_best_device(str(choice))

    # ------------------------------------------------------------------------------
    # data plumbing
    # ------------------------------------------------------------------------------
    def _read(self, data: Any, min_rows: int = 2) -> TelemetryTable:
        """Read new data with the column choices this model was fitted with, wherever the data has those columns."""
        if isinstance(data, TelemetryTable):
            return data
        return load_like(data, sensors=self.sensors, controls=self.controls, time_column=self.time_column,
                         label_column=self.label_column, min_rows=min_rows)

    def _aligned(self, table: TelemetryTable) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """This model's sensor and control columns out of `table` (by name), and which sensor cells were missing."""
        notes: List[str] = []
        positional = (
            table.n_sensors == len(self.sensors)
            and all(name.startswith("sensor_") for name in table.sensors)
            and not set(self.sensors) & set(table.sensors)
        )
        cols, gaps = [], []
        nothing = np.zeros(table.n_rows, dtype=bool)
        for i, name in enumerate(self.sensors):
            if name in table.sensors:
                j = table.sensors.index(name)
                cols.append(table.values[:, j])
                gaps.append(table.missing[:, j])
            elif name in table.controls:
                cols.append(table.control_values[:, table.controls.index(name)])
                gaps.append(nothing)
            elif positional:
                cols.append(table.values[:, i])
                gaps.append(table.missing[:, i])
            else:
                have = ", ".join(table.sensors[:10]) + ("..." if table.n_sensors > 10 else "")
                raise DataError(
                    f"Sensor '{name}' that this model was trained on is missing from the data. "
                    f"Columns found: {have}."
                )
        values = np.stack(cols, axis=1).astype(float)
        missing = np.stack(gaps, axis=1) | ~np.isfinite(values)       # a table edited by hand may hold NaN or inf
        ctrl_cols = []
        missing_ctrl = []
        for name in self.controls:
            if name in table.controls:
                ctrl_cols.append(table.control_values[:, table.controls.index(name)])
            elif name in table.sensors:
                ctrl_cols.append(table.values[:, table.sensors.index(name)])
            else:
                ctrl_cols.append(np.full(table.n_rows, np.nan))
                missing_ctrl.append(name)
        if missing_ctrl:
            notes.append(f"Control column(s) not found, assumed to stay at their typical value: {', '.join(missing_ctrl)}.")
        controls = np.stack(ctrl_cols, axis=1).astype(float) if ctrl_cols else np.zeros((table.n_rows, 0))
        controls = np.where(np.isfinite(controls), controls, np.nan)
        return values, controls, missing, notes

    def _normalise(self, values: np.ndarray, controls: np.ndarray, previous: Optional[np.ndarray] = None,
                   previous_controls: Optional[np.ndarray] = None, gap: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Level-normalised q, its step change dq, standardised velocity p, normalised controls u
        and their standardised step change du. `gap` counts the missing readings just before
        each cell: the usual change is then expected that many more times over.
        """
        qn = (values - self._mean) / self._std
        first = qn[:1] if previous is None else previous.reshape(1, -1)
        dq = np.diff(qn, axis=0, prepend=first)
        usual = self._vmean if gap is None else (gap + 1.0) * self._vmean
        p = (dq - usual) / self._vs                # standardised velocity: 0 means "the usual change"
        if previous is None:
            p[0] = 0.0                              # no previous row: assume the usual change
        if controls.shape[1]:
            un = (np.where(np.isnan(controls), self._cmean, controls) - self._cmean) / self._cstd
            first_u = un[:1] if previous_controls is None else previous_controls.reshape(1, -1)
            du = np.diff(un, axis=0, prepend=first_u) / self._cds
        else:
            un = np.zeros((values.shape[0], 0))
            du = np.zeros((values.shape[0], 0))
        return qn, dq, p, un, du

    def _states(self, qn: np.ndarray, p: np.ndarray) -> np.ndarray:
        """What the engine sees of the sensors: [level (where it means something), velocity]."""
        return np.concatenate([np.clip(qn, -_CLIP, _CLIP) * self._level_mask, np.clip(p, -_CLIP, _CLIP) * self._active], axis=-1)

    def _actions(self, un: np.ndarray, du: np.ndarray) -> np.ndarray:
        """
        The engine's control input for every transition row t -> row t+1, as [level, change].

        A command is taken from row t+1, with the step it makes there: it is known before the
        reading it acts on. Any other control gives its level at row t only. It may be reacting
        to the sensors, so nothing from row t+1 is used; and its latest change is left out as
        well, because a forecast has to hold such a control still, which the engine would then
        read as "everything has stopped moving".
        """
        T, m = un.shape
        if m == 0:
            return np.zeros((T, 1))
        nxt = np.minimum(np.arange(T) + 1, T - 1)
        ahead = self._shift[None, :] > 0
        level = np.where(ahead, un[nxt], un)
        change = np.where(ahead, du[nxt], 0.0)
        if T:
            change[-1] = 0.0                                             # nothing is known to change after the last row
        return np.concatenate([np.clip(level, -_CLIP, _CLIP), np.clip(change, -_CLIP, _CLIP)], axis=1)

    def _future_actions(self, un_last: np.ndarray, du_last: np.ndarray, plan: Optional[np.ndarray], steps: int) -> np.ndarray:
        """
        Control inputs for `steps` transitions past the last observed row, for a batch of
        starting points: un_last, du_last [B, m]; plan [B, steps, m] of normalised control
        levels, or None to hold every control where it is. Returns [B, steps, d_action].
        """
        B, m = un_last.shape
        if m == 0:
            return np.zeros((B, steps, 1))
        future = np.repeat(un_last[:, None, :], steps, axis=1) if plan is None else plan
        levels = np.concatenate([un_last[:, None, :], future], axis=1)                    # rows T .. T+steps
        changes = np.concatenate([du_last[:, None, :], np.diff(levels, axis=1) / self._cds], axis=1)
        ahead = self._shift[None, None, :] > 0
        level = np.where(ahead, levels[:, 1:], levels[:, :-1])
        change = np.where(ahead, changes[:, 1:], 0.0)
        return np.concatenate([np.clip(level, -_CLIP, _CLIP), np.clip(change, -_CLIP, _CLIP)], axis=-1)

    def _tensors(self, qn: np.ndarray, p: np.ndarray, un: np.ndarray, du: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(self._states(qn, p), dtype=torch.float32, device=self._device)
        a = torch.as_tensor(self._actions(un, du), dtype=torch.float32, device=self._device)
        return x, a

    @torch.no_grad()
    def _predict_frozen(self, x: torch.Tensor, a: torch.Tensor,
                        standing_field: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """Leak-free next-step velocity prediction for every row of (x, a), without adapting."""
        n = len(self.sensors)
        outputs = []
        field = standing_field
        for start in range(0, x.shape[0], _CHUNK):
            out = self.engine(x[start:start + _CHUNK].unsqueeze(0), a[start:start + _CHUNK].unsqueeze(0),
                              standing_field=field, plastic_runtime=False)
            field = out["standing_field"]
            outputs.append(out["output"][0, :, n:2 * n])
        if not outputs:
            return np.zeros((0, n)), field
        return torch.cat(outputs, dim=0).double().cpu().numpy(), field

    def _control_changes(self, du: np.ndarray) -> np.ndarray:
        """[T, m] True where a control changed by much more than it usually does from one row to the next."""
        if du.shape[1] == 0:
            return np.zeros(du.shape, dtype=bool)
        return np.abs(du) > self._cnote[None, :]

    # ------------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------------
    def fit(
        self,
        data: Any,
        *,
        sensors: Optional[Sequence[str]] = None,
        controls: Optional[Sequence[str]] = None,
        commands: Union[None, bool, str, Sequence[str]] = None,
        time_column: Optional[str] = None,
        label_column: Optional[str] = None,
        progress: Optional[ProgressFn] = None,
    ) -> "TelemetryModel":
        """
        Learn normal behaviour from a telemetry log. Returns `self`, so calls can be chained.

        `data` may be a file path (CSV/TSV/JSON/JSONL), a list of row dicts, a dict of columns,
        a NumPy array, a pandas DataFrame or a `TelemetryTable`. Train on a period of normal
        operation when you can; a handful of wild readings in the log does not disturb the
        scales the model learns. Label columns are never used for training.

        commands : which control columns are commands, i.e. values decided independently of
            the sensors (setpoints, schedules, operator inputs). The model is told a command's
            value for the reading it is predicting, so it can learn how the sensors respond to
            it; any other control is only used up to the previous reading, because a
            controller output that reacts to the sensors would otherwise give the answer away.
            Default: controls named like setpoints (`setpoint_*`, `sp_*`, `*_setpoint`,
            `*_sp`). Pass a list of names, True / "all", or False / "none". With commands,
            training runs twice: the second model is not told them and makes the forecasts
            for which no plan is given.
        """
        table = load_table(data, sensors=sensors, controls=controls, time_column=time_column, label_column=label_column)
        notes = list(table.notes)
        warnings: List[str] = []
        T, n = table.n_rows, table.n_sensors
        if T < 12:
            raise DataError(f"Training needs at least 12 time steps; this log has {T}.")
        report = progress or (lambda stage, fraction, message: None)
        share = [1.0]                                      # of the progress bar; half when a companion is trained as well

        def say(stage: str, fraction: float, message: str) -> None:
            report(stage, share[0] * fraction, message)

        say("prepare", 0.0, f"Preparing {T:,} rows x {n} sensors")

        cfg = self.settings
        tiny = T < 80
        n_val = 0 if tiny else max(int(round(T * cfg["validation_fraction"])), 0)
        if n_val and n_val < 16:
            n_val = 0
        n_tr = T - n_val
        if tiny:
            warnings.append(f"Very short log ({T} rows): the model is calibrated on its own training data, "
                            "so treat scores and forecasts as indicative only.")
        if table.labels is not None and int(table.labels.sum()):
            notes.append(f"Label column '{table.label_column}' was ignored during training "
                         f"({int(table.labels.sum())} labelled anomalous rows are part of the training log).")

        self.sensors, self.controls = [str(s) for s in table.sensors], [str(c) for c in table.controls]
        self.time_column, self.label_column = table.time_column, table.label_column
        values, ctrl, missing = table.values, table.control_values, np.asarray(table.missing, dtype=bool)
        m = len(self.controls)

        if commands is None:
            self.commands = [c for c in self.controls if _is_setpoint_name(c)]
        elif commands is True or (isinstance(commands, str) and commands.strip().lower() == "all"):
            self.commands = list(self.controls)
        elif commands is False or (isinstance(commands, str) and commands.strip().lower() in ("none", "")):
            self.commands = []
        else:
            wanted = [commands] if isinstance(commands, str) else [str(c) for c in commands]
            unknown = [c for c in wanted if c not in self.controls]
            if unknown:
                raise DataError(f"commands must be control columns; not among the controls ({', '.join(self.controls) or 'none'}): "
                                f"{', '.join(unknown)}.")
            self.commands = [c for c in self.controls if c in wanted]
        self._shift = np.array([1.0 if c in self.commands else 0.0 for c in self.controls])
        if self.commands:
            share[0] = 0.5

        # ---- scales (from the training part only, with wild readings set aside) -----------
        train = values[:n_tr]
        train_c = _clipped(train)
        wild = train != train_c
        if wild.any():
            worst = sorted(((int(wild[:, i].sum()), s) for i, s in enumerate(self.sensors)), reverse=True)[:3]
            notes.append(f"{int(wild.sum())} extreme reading(s) were set aside when learning each sensor's scale: "
                         + ", ".join(f"{s} ({k})" for k, s in worst if k) + ".")
        self._mean = train_c.mean(axis=0)
        std = train_c.std(axis=0)
        constant = std <= 1e-10 * np.abs(train_c).max(axis=0)
        # A sensor that never varied gets a scale on which any visible change is enormous,
        # whatever its units: "3.0 became 3.2" is as much a change as "3000 became 3200".
        self._std = np.where(constant, np.maximum(1e-6 * np.abs(self._mean), 1e-9), std)
        self._active = (~constant).astype(float)
        steps_c = _clipped(np.diff((train_c - self._mean) / self._std, axis=0))
        vs = steps_c.std(axis=0) if steps_c.shape[0] else np.zeros(n)
        self._vmean = steps_c.mean(axis=0) if steps_c.shape[0] else np.zeros(n)   # the usual per-step change (non-zero for counters and trends)
        # A counter that moved by exactly the same amount in every row has no spread to measure a
        # surprise against. Like a constant sensor, it gets a scale on which any other step is enormous.
        steady = (vs < 1e-9) & ~constant
        self._vs = np.where(steady, np.maximum(1e-6 * np.abs(self._vmean), 1e-9), np.where(vs < 1e-9, 1.0, vs))
        if steady.any():
            notes.append("Sensor(s) that changed by exactly the same amount in every training row (any other step will be flagged): "
                         + ", ".join(s for s, c in zip(self.sensors, steady) if c) + ".")
        if m:
            ctrl_c = _clipped(ctrl[:n_tr])
            self._cmean = ctrl_c.mean(axis=0)
            cstd = ctrl_c.std(axis=0)
            self._cstd = np.where(cstd <= 1e-10 * np.maximum(np.abs(ctrl_c).max(axis=0), 1e-300), 1.0, cstd)
            moves = np.diff((ctrl_c - self._cmean) / self._cstd, axis=0)
            # Scale of a control's step change. Setpoints move in occasional steps that dwarf their
            # row-to-row jitter, and the size of a step matters: the largest one in the log is put
            # at half the input range, so that steps up to twice that size can still be told apart.
            kept = _clipped(moves) if moves.shape[0] else np.zeros((1, m))
            spread = np.maximum(kept.std(axis=0), np.abs(kept).max(axis=0) / (0.5 * _CLIP))
            self._cds = np.where(spread < 1e-9, 1.0, spread)
            typical = 1.4826 * np.median(np.abs(moves - np.median(moves, axis=0)), axis=0) if moves.shape[0] else np.zeros(m)
            raw_std = moves.std(axis=0) if moves.shape[0] else np.zeros(m)
            self._cnote = np.maximum(6.0 * np.maximum(typical, 0.1 * raw_std), 1e-9) / self._cds
        else:
            self._cmean, self._cstd, self._cds, self._cnote = np.zeros(0), np.ones(0), np.ones(0), np.ones(0)
        self._gain = np.ones(n)
        if constant.any():
            notes.append("Sensor(s) constant during training (any change will be flagged): "
                         + ", ".join(s for s, c in zip(self.sensors, constant) if c) + ".")

        # ---- which sensors have a stable operating envelope? ------------------------------
        # Counters, totalisers and random-walk-like signals keep leaving the range they were
        # trained on. For those, only the rate of change is modelled: their absolute level is
        # hidden from the engine (so it cannot extrapolate) and the range check is switched off.
        qn_env = (_clipped(values) - self._mean) / self._std
        stable, settled = _envelope(qn_env, n_tr)
        self._level_mask = (stable & ~constant).astype(float)
        free = [s for s, ok in zip(self.sensors, stable) if not ok]
        if free:
            notes.append("Free-running sensor(s) (trend or random walk): only their rate of change is modelled, "
                         "and the range check is off: " + ", ".join(free) + ".")

        qn, dq, p, un, du = self._normalise(values, ctrl)

        # ---- build the engine ------------------------------------------------------------
        d_model = cfg["d_model"]
        if d_model == "auto":
            d_model = 64 if n <= 24 else int(min(256, 32 * math.ceil(2.5 * n / 32)))
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(cfg["seed"])
        try:
            device = self._resolve_device()
            try:
                history = self._train(device, d_model, qn, p, un, du, n_tr, tiny, say)
            except RuntimeError as exc:
                if device.type == "cpu":
                    raise
                warnings.append(f"Training on {device.type} failed ({str(exc).splitlines()[0][:120]}); used the CPU instead.")
                torch.manual_seed(cfg["seed"])
                history = self._train(torch.device("cpu"), d_model, qn, p, un, du, n_tr, tiny, say)
        finally:
            torch.random.set_rng_state(rng_state)

        # ---- calibrate on held-out data ------------------------------------------------
        say("calibrate", 0.9, "Calibrating scores and forecast intervals")
        x, a = self._tensors(qn, p, un, du)
        p_hat_next, _ = self._predict_frozen(x[:-1], a[:-1])          # prediction of p at rows 1..T-1
        target = p[1:]
        real = ~(missing[1:] | missing[:-1])                          # transitions between two real readings
        real[0] = False                                               # row 0 has no velocity yet: row 1 is predicted from a guess
        held_out = slice(0, T - 1) if n_val == 0 else slice(n_tr - 1, T - 1)
        ph, tg, ok = p_hat_next[held_out], np.clip(target[held_out], -_TARGET_CLIP, _TARGET_CLIP), real[held_out]
        num, den = (ph * tg * ok).sum(axis=0), (ph * ph * ok).sum(axis=0)
        gain = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
        self._gain = np.clip(gain, 0.0, 1.0 if n_val == 0 else 1.5)
        counted = np.maximum(ok.sum(axis=0), 1)
        base = (tg ** 2 * ok).sum(axis=0) / counted
        error = ((tg - self._gain * ph) ** 2 * ok).sum(axis=0) / counted
        with np.errstate(divide="ignore", invalid="ignore"):
            sensor_skill = np.where((base > 1e-12) & ~constant, 1.0 - error / base, np.nan)
        skill = float(np.nanmean(sensor_skill)) if np.isfinite(sensor_skill).any() else float("nan")

        # Residual scale. Three things can each raise it, and it is the largest that counts:
        # the held-out residuals (their MAD and their tail); the tail of the residuals over the
        # whole log, so that a kind of jump the log shows in 1% of its rows is not an alarm; and
        # half the scale over the whole log, in case the held-out stretch was unusually quiet
        # (a sensor that sat still there would otherwise alarm at every later movement).
        # The floor says "no model is credibly more than 20x (3x for a log too short to hold
        # data back) better than the naive baseline".
        floor = 0.3 if n_val == 0 else 0.05
        resid_all = np.where(real, target - self._gain * p_hat_next, np.nan)
        mad_held, tail_held = robust_sigma(resid_all[held_out], floor=floor, parts=True)
        mad_all, tail_all = robust_sigma(resid_all, floor=floor, parts=True)
        sigma = np.maximum.reduce([mad_held, tail_held, tail_all, 0.5 * mad_all])
        # Status and mode flags take a few distinct values and switch without warning. No model
        # can say when, so a switch as large as the log shows at least three times is normal.
        flags = []
        if n_tr >= 50:
            for i in range(n):
                if constant[i] or np.unique(train_c[:, i]).size > 6:
                    continue
                seen = np.abs(resid_all[:, i])
                seen = seen[np.isfinite(seen)]
                if seen.size >= 3:
                    sigma[i] = max(sigma[i], float(np.partition(seen, seen.size - 3)[seen.size - 3]) / 4.0)
                    flags.append(self.sensors[i])
        if flags:
            notes.append("Sensor(s) with only a few distinct values (status or mode flags): a switch that the log "
                         "shows at least three times is treated as normal: " + ", ".join(flags) + ".")
        lam = _EWMA_LAMBDA
        # The drift chart gets the same three-way treatment: a model that is slightly but
        # systematically off in some part of a cycle must not ring it all through that part.
        charted = ewma(np.clip(np.nan_to_num(resid_all, nan=0.0), -_TARGET_CLIP * sigma, _TARGET_CLIP * sigma), lam)
        charted[:2] = 0.0                                             # the first rows are not scored either
        slow = floor * math.sqrt(lam / (2 - lam))
        chart_mad_held, chart_tail_held = robust_sigma(charted[held_out], floor=slow, parts=True)
        chart_mad_all, chart_tail_all = robust_sigma(charted, floor=slow, parts=True)
        ewma_sigma = np.maximum.reduce([chart_mad_held, chart_tail_held, chart_tail_all, 0.5 * chart_mad_all,
                                        sigma * math.sqrt(lam / (2 - lam))])
        if n_val == 0:
            sigma, ewma_sigma = sigma * 1.5, ewma_sigma * 1.5

        lo, hi = qn_env.min(axis=0), qn_env.max(axis=0)
        width = hi - lo
        # How far outside the envelope counts as an alarm: a quarter of its width, or 0.4 of it
        # while the envelope was still widening as the log went on.
        margin = np.where(width > 0, np.maximum(np.where(settled, 0.25, 0.4) * width, 0.25), 1.0)
        # An envelope learned from a short log is too narrow to be trusted as a limit. A sensor
        # that never varied is the exception: its "envelope" is a single value.
        range_enabled = np.where(constant, 1.0, stable.astype(float) * (1.0 if n_tr >= 200 else 0.0))

        abs_dq = np.abs(dq[1:])
        flat_eps = np.zeros(n)
        flat_run = np.zeros(n)
        for i in range(n):
            moved = abs_dq[:, i][real[:, i]]
            moving = moved[moved > 0]
            if constant[i] or moving.size == 0:
                continue                                   # constant sensor: the range check covers it
            flat_eps[i] = 1e-3 * float(np.median(moving))
            longest = longest_run(moved <= flat_eps[i])
            if longest <= (T - 1) / 5:
                flat_run[i] = max(8, math.ceil(2.5 * longest) + 2)

        self._cal = ScoreCalibration(sigma=sigma, ewma_sigma=ewma_sigma, ewma_lambda=lam, range_lo=lo, range_hi=hi,
                                     range_margin=margin, range_enabled=range_enabled, flat_eps=flat_eps, flat_run=flat_run)

        # ---- backtest multi-step forecasts ----------------------------------------------
        cal_start = 0 if n_val == 0 else n_tr
        self._fc = self._calibrate_forecasts(qn, p, un, du, cal_start, T)
        forecast_skill: Dict[str, float] = {}
        horizon = int(self._fc["weights"].shape[0]) if self._fc["backtested"] else 0
        if horizon:
            with np.errstate(invalid="ignore"):
                overall = np.nanmean(np.where(constant[None, :], np.nan, self._fc["skill"]), axis=1)
            for h in (1, 6, 12, 24, 48, 96):
                if h <= horizon and np.isfinite(overall[h - 1]):
                    forecast_skill[str(h)] = float(overall[h - 1])

        # ---- remember the end of the log so forecasts can continue from it ------------
        keep = min(T, 3 * _CONTEXT)
        self._tail, self._tail_offset = table.slice(T - keep, None), T - keep
        self._fit_table = table

        weak = [s for s, k in zip(self.sensors, sensor_skill) if np.isfinite(k) and k < 0.02]
        if np.isfinite(skill) and skill < 0.02:
            warnings.append("The model predicts the next reading no better than repeating the current one. "
                            "Anomaly scoring still works, but expect little from forecasts.")
        elif weak and len(weak) < n:
            notes.append("No predictive skill beyond 'last value' for: " + ", ".join(weak) + ".")

        # ---- forecasts for which no plan is given ----------------------------------------
        # A model that is told its commands learns to lean on them, and a forecast made without
        # a plan can only tell it that they stay where they are. A companion that is never told
        # them carries on from the sensors' own behaviour instead, exactly as a model without
        # commands does, so declaring commands cannot make such a forecast worse.
        self._free = None
        seconds = history["seconds"]
        if self.commands:
            companion = TelemetryModel(**cfg)
            companion.fit(table, commands=False,
                          progress=lambda stage, fraction, message: report("companion", 0.5 + 0.49 * fraction,
                                                                           "Forecasts without a plan: " + message[:1].lower() + message[1:]))
            self._free = companion
            card = companion.fit_summary_
            forecast_skill, horizon, seconds = dict(card.forecast_skill), card.forecast_horizon, seconds + card.seconds
            notes.append("Forecasts for which no plan is given come from a companion model that is not told the commands "
                         "in advance, so they do not assume the commands stay where they are. Training ran twice for this.")
            share[0] = 1.0

        self.fit_summary_ = FitSummary(
            rows=T, train_rows=n_tr, validation_rows=n_val, sensors=list(self.sensors), controls=list(self.controls),
            steps_run=history["steps"], best_step=history["best_step"], seconds=seconds,
            device=describe_accelerator(self._device),
            parameters=sum(prm.numel() for prm in self.engine.parameters()),
            skill=skill, sensor_skill={s: float(k) for s, k in zip(self.sensors, sensor_skill)},
            forecast_skill=forecast_skill, forecast_horizon=horizon,
            warnings=warnings, notes=notes, source=table.source, commands=list(self.commands),
        )
        say("done", 1.0, "Model ready")
        return self

    def _train(self, device: torch.device, d_model: int, qn: np.ndarray, p: np.ndarray, un: np.ndarray, du: np.ndarray,
               n_tr: int, tiny: bool, say: ProgressFn) -> Dict[str, Any]:
        cfg = self.settings
        n = qn.shape[1]
        self._device = device
        engine = ResonantCausalWorldEngine(
            d_input=2 * n, d_model=int(d_model), d_action=max(2 * un.shape[1], 1),
            num_worldlines=cfg["num_worldlines"], num_freq_bands=cfg["num_freq_bands"],
        ).to(device)
        self.engine = engine
        x, a = self._tensors(qn, p, un, du)
        x_in, a_in, y_out = x[:-1], a[:-1], x[1:]
        T = x_in.shape[0]
        train_end = n_tr - 1                               # transitions [0, train_end) are for training
        L = max(2, min(cfg["window"], train_end))
        B = cfg["batch_size"]
        max_steps = min(cfg["max_steps"], 150) if tiny else cfg["max_steps"]
        eval_every = 25
        has_val = train_end < T

        opt = torch.optim.AdamW(engine.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_steps)
        gen = torch.Generator().manual_seed(cfg["seed"])
        best_loss, best_state, best_step, stale = float("inf"), None, 0, 0
        ctx_lo = max(train_end - 64, 0)
        started = time.perf_counter()
        step = 0
        for step in range(1, max_steps + 1):
            engine.train()
            starts = torch.randint(0, max(train_end - L, 0) + 1, (B,), generator=gen).tolist()
            xb = torch.stack([x_in[s:s + L] for s in starts])
            ab = torch.stack([a_in[s:s + L] for s in starts])
            yb = torch.stack([y_out[s:s + L] for s in starts])
            opt.zero_grad()
            out = engine(xb, ab, plastic_runtime=False)
            branches = out["branch_predictions"]
            target_p = yb[..., n:2 * n]
            loss = (
                F.mse_loss(out["output"][..., n:2 * n], target_p)
                + 0.5 * F.mse_loss(branches[..., n:2 * n], target_p.unsqueeze(0).expand(branches.shape[0], -1, -1, -1))
                + 0.25 * F.mse_loss(out["output"], yb)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Training diverged (loss became non-finite). Check the data for extreme values.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.parameters(), 2.0)
            opt.step()
            sched.step()

            if step % eval_every == 0 or step == max_steps:
                engine.eval()
                with torch.no_grad():
                    if has_val:
                        pred = engine(x_in[ctx_lo:].unsqueeze(0), a_in[ctx_lo:].unsqueeze(0), plastic_runtime=False)["output"][0]
                        offset = train_end - ctx_lo
                        val = F.mse_loss(pred[offset:, n:2 * n], y_out[train_end:, n:2 * n]).item()
                    else:
                        pred = engine(x_in.unsqueeze(0), a_in.unsqueeze(0), plastic_runtime=False)["output"][0]
                        val = F.mse_loss(pred[:, n:2 * n], y_out[:, n:2 * n]).item()
                if val < best_loss - 1e-6:
                    best_loss, best_step, stale = val, step, 0
                    best_state = copy.deepcopy(engine.state_dict())
                else:
                    stale += eval_every
                say("train", 0.05 + 0.85 * step / max_steps, f"Training step {step}/{max_steps} (validation error {val:.4f})")
                if has_val and stale >= cfg["patience"]:
                    break
        if best_state is not None:
            engine.load_state_dict(best_state)
        engine.eval()
        engine.reset_runtime_adaptation()
        return {"steps": step, "best_step": best_step, "seconds": time.perf_counter() - started}

    # ------------------------------------------------------------------------------
    # forecasting internals
    # ------------------------------------------------------------------------------
    @torch.no_grad()
    def _rollout(self, q0: np.ndarray, p0: np.ndarray, field: Optional[torch.Tensor], actions: np.ndarray) -> np.ndarray:
        """Free-run the engine over `actions` [B, H, d_action] from states (q0, p0) [B, n]; returns level-normalised q-hat [B, H, n]."""
        n = len(self.sensors)
        dev = self._device
        q = torch.as_tensor(q0, dtype=torch.float32, device=dev)
        p = torch.as_tensor(p0, dtype=torch.float32, device=dev)
        u = torch.as_tensor(actions, dtype=torch.float32, device=dev)
        vs = torch.as_tensor(self._vs, dtype=torch.float32, device=dev)
        vmean = torch.as_tensor(self._vmean, dtype=torch.float32, device=dev)
        gain = torch.as_tensor(self._gain, dtype=torch.float32, device=dev)
        mask = torch.as_tensor(self._level_mask, dtype=torch.float32, device=dev)
        active = torch.as_tensor(self._active, dtype=torch.float32, device=dev)
        steps = []
        for h in range(u.shape[1]):
            x = torch.cat([q.clamp(-_CLIP, _CLIP) * mask, p.clamp(-_CLIP, _CLIP) * active], dim=-1).unsqueeze(1)
            out = self.engine(x, u[:, h:h + 1, :], standing_field=field, plastic_runtime=False)
            field = out["standing_field"]
            p = out["output"][:, 0, n:2 * n] * gain * active
            q = q + (vmean + vs * p) * active
            steps.append(q)
        return torch.stack(steps, dim=1).double().cpu().numpy()

    @torch.no_grad()
    def _warm_field(self, x: torch.Tensor, a: torch.Tensor, ends: Sequence[int], context: int = 64) -> Optional[torch.Tensor]:
        """Standing wavefields after reading the `context` rows that precede each index in `ends`."""
        length = min(context, min(ends))
        if length < 1:
            return None
        xb = torch.stack([x[e - length:e] for e in ends])
        ab = torch.stack([a[e - length:e] for e in ends])
        return self.engine(xb, ab, plastic_runtime=False)["standing_field"]

    def _calibrate_forecasts(self, qn: np.ndarray, p: np.ndarray, un: np.ndarray, du: np.ndarray, lo: int, hi: int) -> Dict[str, Any]:
        """Backtest free-running forecasts on rows [lo, hi) to learn trust weights and error quantiles."""
        n = qn.shape[1]
        span = hi - lo
        H = int(min(self.settings["forecast_horizon"], (span - 8) // 2))
        one_step = self._cal.sigma * self._vs               # typical one-step error, level-normalised
        if H < 4:
            return {"backtested": False, "weights": np.ones((1, n)), "q80": 1.2816 * one_step[None, :],
                    "q95": 1.96 * one_step[None, :], "skill": np.full((1, n), np.nan)}
        last_anchor = hi - H - 1
        first_anchor = max(lo, 1)
        stride = max(1, (last_anchor - first_anchor) // 64)
        anchors = list(range(first_anchor, last_anchor + 1, stride))
        x, a = self._tensors(qn, p, un, du)
        field = self._warm_field(x, a, anchors)
        held = self._future_actions(un[anchors], du[anchors], None, H)   # controls unknown in advance: hold the last value
        rolled = self._rollout(qn[anchors], p[anchors], field, held)
        # The naive reference is "keep changing at the usual rate" (plain last-value for stationary sensors).
        naive = qn[anchors][:, None, :] + np.arange(1, H + 1)[None, :, None] * self._vmean[None, None, :]
        truth = np.stack([qn[t + 1:t + 1 + H] for t in anchors])
        d_model, d_true = rolled - naive, truth - naive
        num, den = (d_model * d_true).sum(axis=0), (d_model ** 2).sum(axis=0)
        weights = np.clip(np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0), 0.0, 1.0)
        if H >= 5:                                           # smooth the trust weights across horizons
            padded = np.pad(weights, ((2, 2), (0, 0)), mode="edge")
            weights = np.stack([padded[i:i + H] for i in range(5)]).mean(axis=0)
        err = truth - (naive + weights[None] * d_model)
        inflate = 1.25 if len(anchors) < 8 else 1.0
        q80 = np.maximum.accumulate(np.quantile(np.abs(err), 0.80, axis=0), axis=0) * inflate
        q95 = np.maximum.accumulate(np.quantile(np.abs(err), 0.95, axis=0), axis=0) * inflate
        q80 = np.maximum(q80, 1.2816 * one_step[None, :])
        q95 = np.maximum(q95, np.maximum(1.96 * one_step[None, :], q80))
        base = (d_true ** 2).mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            skill = np.where(base > 1e-12, 1.0 - (err ** 2).mean(axis=0) / base, np.nan)
        return {"backtested": True, "weights": weights, "q80": q80, "q95": q95, "skill": skill}

    # ------------------------------------------------------------------------------
    # anomaly detection
    # ------------------------------------------------------------------------------
    def detect(
        self,
        data: Any = None,
        *,
        sensitivity: Union[str, float] = "medium",
        adapt: bool = False,
        merge_gap: int = 3,
        min_duration: int = 1,
        range_check: Optional[bool] = None,
        progress: Optional[ProgressFn] = None,
    ) -> DetectionResult:
        """
        Score a telemetry log and group anomalous stretches into explained events.

        Parameters
        ----------
        data : same inputs as `fit`. Omit it to scan the log the model was just trained on.
            Columns are read the way they were for `fit` (the same sensors, controls, time
            and label columns) wherever the data has them.
        sensitivity : "low" | "medium" | "high", or the score threshold in sigmas (default 4.5).
        adapt : False (default) compares every reading with the behaviour learned in `fit`, so
            an abnormal period stays flagged for as long as it lasts. True switches on the
            engine's online plasticity: the model re-learns as the log is replayed, so it flags
            the *onset* of a change and then accepts the new normal.
        merge_gap : exceedances separated by at most this many quiet steps form one event.
        min_duration : minimum number of threshold exceedances for an event to be reported.
        range_check : also flag readings well outside the range seen in training (novelty check).
            On by default in reference mode and off in adaptive mode (where settling at a new
            level is exactly what should be accepted). Turn it off if operating points you
            have not trained on are expected and fine.
        """
        self._require_fitted()
        if data is None:
            if self._fit_table is None:
                raise DataError("No data given. Pass a file or table to detect(); a loaded model does not carry its training log.")
            table = self._fit_table
            extra_notes = ["Scanned the training log itself: anomalies the model was trained on can be under-reported."]
        else:
            table = self._read(data)
            extra_notes = []
        if table.n_rows < 3:
            raise DataError("Detection needs at least 3 time steps.")
        threshold = resolve_threshold(sensitivity)
        use_range = (not adapt) if range_check is None else bool(range_check)
        values, ctrl, missing, align_notes = self._aligned(table)
        # A reading that was missing is not scored, and the reading after a gap is compared with
        # the last real one. An interpolated value would already know what comes next.
        held, gap = _hold_gaps(values, missing)
        qn, dq, p, un, du = self._normalise(held, ctrl, gap=gap)
        T, n = qn.shape
        say = progress or (lambda stage, fraction, message: None)

        if adapt:
            monitor = StreamMonitor(self, adapt=True, sensitivity=threshold)
            states, actions = self._states(qn, p), self._actions(un, du)
            p_hat = np.zeros((T, n))
            for t in range(T):
                p_hat[t] = monitor._advance(states[t], actions[t - 1] if t else None)
                if t % 200 == 0:
                    say("detect", t / T, f"Replaying with online adaptation: row {t:,}/{T:,}")
        else:
            x, a = self._tensors(qn, p, un, du)
            p_next, _ = self._predict_frozen(x[:-1], a[:-1])
            p_hat = np.vstack([np.zeros((1, n)), p_next])

        valid = ~missing
        # After a gap of g readings there were g + 1 steps in which to drift: allow for that.
        residual = np.clip((p - self._gain * p_hat) / np.sqrt(gap + 1.0), -_Z_CAP, _Z_CAP)
        # The first row has nothing to be compared with, and the second is predicted from a state
        # whose velocity had to be guessed: neither can surprise. (Range and stuck-sensor checks,
        # which need no prediction, apply from the first row.)
        residual[:2] = 0.0
        moved = dq.copy()
        moved[0] = np.nan                                  # no previous reading: not "unchanged" either
        scored = score_residuals(residual, level=qn, delta=moved, cal=self._cal, valid=valid, range_check=use_range)
        sensor_scores = np.minimum(scored.sensor_score, _Z_CAP)
        scores = np.minimum(scored.score, _Z_CAP)

        expected_qn = qn.copy()
        expected_qn[1:] = qn[:-1] + (gap[1:] + 1.0) * self._vmean + self._vs * self._gain * p_hat[1:]
        expected = expected_qn * self._std + self._mean
        band = threshold * self._cal.sigma * self._vs * self._std

        stamps = [table.time_label(i) for i in range(T)]
        spans = find_events(scores, threshold, merge_gap=merge_gap, min_duration=min_duration)
        changed = self._control_changes(du)
        events = [self._describe_event(k, a_, b_, scored, sensor_scores, scores, threshold, held, expected, stamps, ctrl, changed)
                  for k, (a_, b_) in enumerate(spans, 1)]

        metrics = None
        if table.labels is not None and table.labels.shape[0] == T:
            metrics = detection_metrics(table.labels, scores, spans, tolerance=merge_gap)

        notes = extra_notes + [note for note in table.notes if not note.startswith("Filled ")] + align_notes
        gaps = int(missing.sum())
        if gaps:
            notes.append(f"{gaps} missing reading(s): the last real value was held and those cells were not scored.")
        if adapt:
            notes.append("Adaptive mode: the model kept learning during the replay, so events mark the onset of "
                         "changes rather than their full duration.")
        say("detect", 1.0, f"{len(events)} event(s) found")
        return DetectionResult(
            sensors=list(self.sensors), timestamps=stamps, values=held, expected=expected, band=band,
            residual_z=np.clip(scored.z, -_Z_CAP, _Z_CAP), sensor_scores=sensor_scores, scores=scores,
            threshold=threshold, events=events, mode="adaptive" if adapt else "reference",
            time_seconds=table.time_seconds, labels=table.labels, metrics=metrics, notes=notes,
            source=table.source, model_info=self._model_info(),
        )

    def _describe_event(self, index: int, a: int, b: int, scored: Any, sensor_scores: np.ndarray, scores: np.ndarray,
                        threshold: float, values: np.ndarray, expected: np.ndarray, stamps: List[str],
                        ctrl: np.ndarray, changed: np.ndarray) -> AnomalyEvent:
        seg = sensor_scores[a:b + 1]
        peak = a + int(np.argmax(scores[a:b + 1]))
        excess = np.maximum(seg - threshold, 0.0).sum(axis=0)
        peaks = seg.max(axis=0)
        total = float(excess.sum()) or 1.0
        # Rank by peak evidence: the sensor that misbehaves produces the sharpest deviation, while
        # sensors that merely disagree with it (knock-on effects) build up weaker, slower evidence.
        # A secondary sensor is named when its own evidence is a fair fraction of the lead's -- or is
        # overwhelming in its own right, however extreme the lead is.
        ranked = np.argsort(-peaks, kind="stable")
        floor = max(1.2 * threshold, min(0.3 * float(peaks[ranked[0]]), 5.0 * threshold))
        order = [int(i) for k, i in enumerate(ranked) if peaks[i] > threshold and (k == 0 or peaks[i] >= floor)][:5]
        involved: List[Dict[str, Any]] = []
        brief = b - a + 1 <= 4
        for i in order:
            above = seg[:, i] > threshold
            first = int(np.argmax(above))
            # Describe the deviation in the direction it was *sustained*: for a level shift the
            # single largest residual is often the snap back to normal at the very end.
            sustained = float(np.sum(scored.drift_z[a:b + 1, i][above]))
            if brief or sustained == 0.0:
                t_i = a + first
                signed = float(scored.z[t_i, i]) or sustained
            else:
                same_side = np.sign(scored.z[a:b + 1, i]) == np.sign(sustained)
                pick = np.where(same_side & above, seg[:, i], -np.inf)
                t_i = a + (int(np.argmax(pick)) if np.isfinite(pick).any() else int(np.argmax(seg[:, i])))
                signed = sustained
            reason = REASONS[int(scored.reason[a + int(np.argmax(seg[:, i])), i])]
            outside = scored.range_z[a:b + 1, i] > threshold
            if reason != "flatline" and a <= 1 and outside[0]:
                # The log begins in the middle of this: the onset was never seen, so the largest
                # residual is the jump back to normal. Describe where the sensor was instead.
                reason = "range"
            if reason == "range":
                # Report the most extreme reading on the side it went out of range.
                centre = 0.5 * (self._cal.range_lo[i] + self._cal.range_hi[i]) * self._std[i] + self._mean[i]
                span = values[a:b + 1, i]
                high = float(np.sum(span[outside] > centre)) >= 0.5 * max(int(outside.sum()), 1)
                t_i = a + int(np.argmax(span) if high else np.argmin(span))
                signed = 1.0 if high else -1.0
            elif reason == "flatline":
                t_i = a + int(np.argmax(seg[:, i]))
            involved.append({
                "name": self.sensors[i], "peak_score": float(peaks[i]), "share": float(excess[i] / total),
                "reason": reason, "direction": "above" if signed >= 0 else "below",
                "peak_time": stamps[t_i], "actual": float(values[t_i, i]), "expected": float(expected[t_i, i]),
            })
        lead_reason = involved[0]["reason"] if involved else "surprise"
        if lead_reason == "flatline":
            kind = "flatline"
        elif lead_reason == "range":
            kind = "out of range"
        elif brief:
            kind = "spike"
        else:
            kind = "sustained deviation"
        peak_score = float(scores[peak])

        # A control input that changed markedly as the event began, or just before, is the first
        # thing to look at. (A change after it began is more likely a reaction than a cause.)
        control_changes: List[Dict[str, Any]] = []
        for j, name in enumerate(self.controls):
            for t in range(a, max(a - 4, 0), -1):
                if changed[t, j]:
                    control_changes.append({"name": name, "time": stamps[t], "before": float(ctrl[t - 1, j]), "after": float(ctrl[t, j])})
                    break

        when = stamps[a] if a == b else f"{stamps[a]} to {stamps[b]}"
        length = "1 step" if a == b else f"{b - a + 1} steps"
        if involved:
            lead = involved[0]
            actual, wanted = _fmt_pair(lead["actual"], lead["expected"])
            if lead["reason"] == "flatline":
                what = f"{lead['name']} stopped changing (stuck at {_fmt(lead['actual'])})"
            elif lead["reason"] == "range":
                what = (f"{lead['name']} went {lead['direction']} the range seen in training "
                        f"(reached {_fmt(lead['actual'])})")
            elif kind == "spike":
                what = f"{lead['name']} jumped {lead['direction']} expectations ({actual} vs {wanted} expected)"
            else:
                what = f"{lead['name']} ran {lead['direction']} expectations ({actual} vs {wanted} expected at the peak)"
            others = [s["name"] for s in involved[1:4]]
            also = f"; also involved: {', '.join(others)}" if others else ""
            finding = f"{what}, peaking at {peak_score:.1f} sigma{also}"
        else:
            finding = f"anomaly score peaked at {peak_score:.1f} sigma"
        if control_changes:
            moved = [f"{c['name']} ({' to '.join(_fmt_pair(c['before'], c['after']))})" for c in control_changes[:3]]
            finding += f"; coincides with a change in {', '.join(moved)}"
        finding += "."
        description = f"{when} ({length}): {finding}"
        return AnomalyEvent(
            index=index, start=a, end=b, start_time=stamps[a], end_time=stamps[b], peak=peak, peak_time=stamps[peak],
            peak_score=peak_score, severity=_severity(peak_score / threshold), kind=kind,
            sensors=involved, description=description, what=finding, control_changes=control_changes,
        )

    def _model_info(self) -> Dict[str, Any]:
        from rcwe import __version__
        info: Dict[str, Any] = {"rcwe_version": __version__, "sensors": len(self.sensors), "controls": len(self.controls)}
        if self.fit_summary_ is not None:
            info.update(self.fit_summary_.to_dict())
        return info

    def stream(self, *, sensitivity: Union[str, float] = "medium", adapt: bool = False) -> "StreamMonitor":
        """Create a monitor that scores readings one row at a time (for live data)."""
        self._require_fitted()
        return StreamMonitor(self, adapt=adapt, sensitivity=sensitivity)

    # ------------------------------------------------------------------------------
    # forecasting
    # ------------------------------------------------------------------------------
    def forecast(self, steps: int = 48, data: Any = None, *, future_controls: Any = None) -> Forecast:
        """
        Forecast every sensor `steps` samples past the end of `data` (or of the training log).

        Prediction intervals are empirical: they come from backtests on held-out data made
        during `fit`. Where the backtest shows no skill over "the value stays where it is",
        the forecast falls back to exactly that -- and says so in `Forecast.model_weight`.

        future_controls : optional planned control values for the forecast window
            (array [steps, n_controls], or a table/file with the control columns). Without
            it, controls are assumed to stay at their latest values, and a model with
            commands answers through its companion, which does not use them as commands.
            A plan moves the forecast only as far as the model has learned to respond to
            that control: see `commands` in `fit`.
        """
        self._require_fitted()
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        if future_controls is None and self._free is not None:
            forecast = self._free.forecast(steps, data)
            forecast.notes = [note for note in forecast.notes if note != _CONTROLS_HELD] + [
                f"No plan was given for {', '.join(self.commands)}, so this forecast does not use them as commands. "
                "Pass future_controls (--planned-controls) to see what a planned change does."]
            return forecast
        notes: List[str] = []
        if data is None:
            if self._tail is None:
                raise DataError("No data given and the model does not remember the end of its training log.")
            table, row_offset = self._tail, self._tail_offset
        else:
            table, row_offset = self._read(data), 0
            if table.n_rows < 2:
                raise DataError("Forecasting needs at least 2 rows of recent history.")
        values, ctrl, missing, align_notes = self._aligned(table)
        notes.extend(align_notes)
        values, _ = _hold_gaps(values, missing)
        if table.time_seconds is not None and table.sample_interval is None and self._tail is not None \
                and self._tail.sample_interval:
            # Too few distinct stamps to tell the pace: use the training log's.
            pace = self._tail.sample_interval
            labels = [format_time(float(table.time_seconds[-1]) + pace * k, table.time_offset) for k in range(1, steps + 1)]
        else:
            labels = table.labels_after(steps, row_offset)

        qn, dq, p, un, du = self._normalise(values[-_CONTEXT:], ctrl[-_CONTEXT:])
        x, a = self._tensors(qn, p, un, du)
        field = None
        if x.shape[0] > 1:
            with torch.no_grad():
                field = self.engine(x[:-1].unsqueeze(0), a[:-1].unsqueeze(0), plastic_runtime=False)["standing_field"]

        plan = None
        if future_controls is not None and self.controls:
            planned = self._planned_controls(future_controls)
            if planned.shape[0] < steps:
                notes.append(f"future_controls covers {planned.shape[0]} of {steps} steps; the last row is held after that.")
                planned = np.vstack([planned, np.repeat(planned[-1:], steps - planned.shape[0], axis=0)])
            plan = ((planned[:steps] - self._cmean) / self._cstd)[None, :, :]
            unheard = [c for c in self.controls if c not in self.commands]
            if unheard:
                notes.append("The model reacts to a plan mainly through controls it was told are commands; "
                             f"it has not learned to anticipate a change in: {', '.join(unheard)}.")
        elif self.controls:
            notes.append(_CONTROLS_HELD)
        elif future_controls is not None:
            notes.append("This model has no control inputs; future_controls was ignored.")

        rolled = self._rollout(qn[-1:], p[-1:], field, self._future_actions(un[-1:], du[-1:], None, steps))[0]    # [steps, n]
        fc = self._fc
        H = fc["weights"].shape[0]
        idx = np.minimum(np.arange(steps), H - 1)
        growth = np.sqrt(np.maximum(np.arange(1, steps + 1) / H, 1.0))[:, None]
        weights = fc["weights"][idx]
        naive = qn[-1][None, :] + np.arange(1, steps + 1)[:, None] * self._vmean[None, :] * self._active[None, :]
        # The trust weights were learned with controls held where they are, so they apply to that
        # forecast. What a plan changes is the model's own statement and is added in full.
        centre = naive + weights * (rolled - naive)
        if plan is not None:
            centre = centre + self._rollout(qn[-1:], p[-1:], field, self._future_actions(un[-1:], du[-1:], plan, steps))[0] - rolled
        q80, q95 = fc["q80"][idx] * growth, fc["q95"][idx] * growth
        if not fc["backtested"]:
            spread = np.sqrt(np.arange(1, steps + 1))[:, None]
            q80, q95 = fc["q80"][idx] * spread, fc["q95"][idx] * spread
            notes.append("Too little held-out data to backtest forecasts; intervals are rough estimates.")
        elif steps > H:
            notes.append(f"Horizons beyond {H} steps were not backtested; intervals there are extrapolated.")

        def unscale(arr: np.ndarray) -> np.ndarray:
            return arr * self._std + self._mean

        future_seconds: Optional[np.ndarray] = None
        pace = table.sample_interval or (self._tail.sample_interval if self._tail is not None else None)
        if table.time_seconds is not None and pace:
            future_seconds = float(table.time_seconds[-1]) + pace * np.arange(1, steps + 1)
        stamps = table.timestamps
        hist = min(values.shape[0], max(3 * steps, 48), 3 * _CONTEXT)
        skill_row = fc["skill"][min(steps, H) - 1]
        return Forecast(
            sensors=list(self.sensors), values=unscale(centre), lower=unscale(centre - q80), upper=unscale(centre + q80),
            lower95=unscale(centre - q95), upper95=unscale(centre + q95), timestamps=labels, time_seconds=future_seconds,
            origin=str(stamps[-1]) if stamps is not None else f"step {row_offset + table.n_rows - 1}",
            history=values[-hist:].copy(),
            history_timestamps=None if stamps is None else [str(s) for s in stamps[-hist:]],
            sensor_skill={s: float(k) for s, k in zip(self.sensors, skill_row)},
            model_weight={s: float(w) for s, w in zip(self.sensors, weights.mean(axis=0))},
            notes=notes,
        )

    def _planned_controls(self, source: Any) -> np.ndarray:
        """
        Planned control values for a forecast window as an array [rows, n_controls].

        `source` is an array (columns in the model's control order), or a file / table / list of
        row dicts that holds the control columns by name. Other columns are ignored, so a file
        with nothing but the controls works as well as a complete log.
        """
        names = ", ".join(self.controls)
        if isinstance(source, (list, tuple)) and source and not isinstance(source[0], dict):
            try:
                source = np.asarray(source, dtype=float)
            except (TypeError, ValueError) as exc:
                raise DataError(f"future_controls must be rows of numbers, one column per control ({names}).") from exc
        if hasattr(source, "detach") and hasattr(source, "cpu"):           # torch.Tensor
            source = source.detach().cpu().numpy()
        if isinstance(source, np.ndarray):
            planned = np.asarray(source, dtype=float)
            if planned.ndim == 1:
                planned = planned.reshape(-1, 1)
            if planned.ndim != 2 or planned.shape[1] != len(self.controls):
                raise DataError(
                    f"future_controls has shape {tuple(np.shape(source))}; expected [steps, {len(self.controls)}] "
                    f"with one column per control ({names})."
                )
        elif isinstance(source, TelemetryTable):
            try:
                planned = np.stack([source.column(c) for c in self.controls], axis=1)
            except KeyError as exc:
                raise DataError(f"future_controls has no column {exc}; it needs the control column(s) {names}.") from exc
        else:
            try:
                plan_table = load_like(source, sensors=list(self.controls), min_rows=1)
            except DataError as exc:
                raise DataError(f"future_controls needs the control column(s) {names}: {exc}") from exc
            if plan_table.sensors != list(self.controls):
                found = ", ".join(plan_table.sensors + plan_table.controls) or "none"
                raise DataError(f"future_controls needs the control column(s) {names}; columns found: {found}.")
            planned = plan_table.values
        if planned.shape[0] == 0:
            raise DataError("future_controls is empty.")
        # A gap in the plan is filled the way a gap in a log is: with the control's typical value.
        return np.where(np.isfinite(planned), planned, self._cmean)

    # ------------------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------------------
    def _payload(self) -> Dict[str, Any]:
        """Everything `save` writes, as plain containers and tensors (no checksum yet)."""

        def tensor(arr: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            return None if arr is None else torch.as_tensor(np.asarray(arr), dtype=torch.float64)

        from rcwe import __version__
        tail = self._tail
        payload = {
            "format": FORMAT_NAME, "format_version": FORMAT_VERSION, "rcwe_version": __version__,
            "settings": dict(self.settings),
            "engine_config": {
                "d_input": self.engine.d_input, "d_model": self.engine.d_model, "d_action": self.engine.d_action,
                "num_worldlines": self.engine.num_worldlines, "num_freq_bands": self.engine.num_freq_bands,
            },
            "engine_state": {k: v.detach().cpu() for k, v in self.engine.state_dict().items()},
            "sensors": [str(s) for s in self.sensors], "controls": [str(c) for c in self.controls],
            "commands": [str(c) for c in self.commands],
            "time_column": None if self.time_column is None else str(self.time_column),
            "label_column": None if self.label_column is None else str(self.label_column),
            "scaler": {"mean": tensor(self._mean), "std": tensor(self._std), "vs": tensor(self._vs),
                       "vmean": tensor(self._vmean),
                       "cmean": tensor(self._cmean), "cstd": tensor(self._cstd), "gain": tensor(self._gain),
                       "cds": tensor(self._cds), "cnote": tensor(self._cnote), "shift": tensor(self._shift),
                       "level_mask": tensor(self._level_mask), "active": tensor(self._active)},
            "calibration": self._cal.to_payload(),
            "forecast": {"backtested": bool(self._fc["backtested"]), "weights": tensor(self._fc["weights"]),
                         "q80": tensor(self._fc["q80"]), "q95": tensor(self._fc["q95"]), "skill": tensor(self._fc["skill"])},
            "fit_summary": None if self.fit_summary_ is None else self.fit_summary_.to_dict(),
            "tail": None if tail is None else {
                "values": tensor(tail.values), "controls": tensor(tail.control_values),
                "timestamps": None if tail.timestamps is None else [str(s) for s in tail.timestamps],
                "time_seconds": tensor(tail.time_seconds), "step_values": tensor(tail.step_values),
                "time_offset": None if tail.time_offset is None else float(tail.time_offset),
                "offset": int(self._tail_offset),
            },
            "companion": None if self._free is None else self._free._payload(),
        }
        return payload

    def save(self, path: Union[str, Path]) -> str:
        """Save everything needed to reuse the model (weights, scalers, calibration) in one file."""
        self._require_fitted()
        payload = self._payload()
        payload["digest"] = _digest(payload)
        out = Path(path)
        if out.is_dir():
            raise DataError(f"{out} is a folder; give a file name for the model.")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)
        return str(out.resolve())

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "auto") -> "TelemetryModel":
        """Load a model written by `save()`. Uses PyTorch's safe `weights_only` loader."""
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"Model file not found: {file}")
        if file.is_dir():
            raise DataError(f"{file} is a folder, not a model file.")
        try:
            payload = torch.load(file, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - surface a readable message for any unpickling problem
            reason = (str(exc).splitlines() or [type(exc).__name__])[0][:160] or type(exc).__name__
            raise DataError(f"{file.name} is not a readable RCWE model file ({reason}).") from exc
        if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
            hint = ""
            if isinstance(payload, dict) and "state_dict" in payload and "config" in payload:
                hint = " It looks like a raw engine checkpoint: open it with ResonantCausalWorldEngine.load_pretrained()."
            raise DataError(f"{file.name} is not an RCWE telemetry model.{hint}")
        version = int(payload.get("format_version", 0))
        if version > FORMAT_VERSION:
            raise DataError(f"{file.name} was written by a newer RCWE version (format {version}); please upgrade.")
        if version < FORMAT_VERSION:
            raise DataError(f"{file.name} was written by an earlier build of RCWE (format {version}) and cannot be read by "
                            "this one. Train the model again with `rcwe fit`.")
        try:
            intact = payload.get("digest") == _digest(payload)
        except Exception:  # noqa: BLE001 - a payload that cannot even be fingerprinted is damaged
            intact = False
        if not intact:
            raise DataError(f"{file.name} is damaged: its contents do not match the checksum written when it was saved.")
        try:
            return cls._from_payload(payload, device)
        except (KeyError, TypeError, AttributeError, RuntimeError, IndexError) as exc:
            reason = (str(exc).splitlines() or [type(exc).__name__])[0][:160] or type(exc).__name__
            raise DataError(f"{file.name} is damaged or incomplete ({type(exc).__name__}: {reason}).") from exc

    @classmethod
    def _from_payload(cls, payload: Dict[str, Any], device: str) -> "TelemetryModel":
        settings = dict(payload["settings"])
        settings["device"] = device
        model = cls(**settings)
        model._device = model._resolve_device()
        cfg = payload["engine_config"]
        engine = ResonantCausalWorldEngine(**{k: int(v) for k, v in cfg.items()})
        engine.load_state_dict(payload["engine_state"])
        model.engine = engine.to(model._device).eval()
        model.sensors, model.controls = list(payload["sensors"]), list(payload["controls"])
        model.commands = list(payload.get("commands", []))
        model.time_column, model.label_column = payload.get("time_column"), payload.get("label_column")

        def array(value: Optional[torch.Tensor]) -> Optional[np.ndarray]:
            return None if value is None else value.double().numpy()

        sc = payload["scaler"]
        model._mean, model._std, model._vs = array(sc["mean"]), array(sc["std"]), array(sc["vs"])
        model._vmean = array(sc["vmean"])
        model._cmean, model._cstd, model._gain = array(sc["cmean"]), array(sc["cstd"]), array(sc["gain"])
        model._cds, model._cnote, model._shift = array(sc["cds"]), array(sc["cnote"]), array(sc["shift"])
        model._level_mask, model._active = array(sc["level_mask"]), array(sc["active"])
        if model._mean.shape[0] != len(model.sensors) or model._cmean.shape[0] != len(model.controls):
            raise KeyError("scalers do not match the column lists")
        model._cal = ScoreCalibration.from_payload(payload["calibration"])
        fc = payload["forecast"]
        model._fc = {"backtested": bool(fc["backtested"]), "weights": array(fc["weights"]), "q80": array(fc["q80"]),
                     "q95": array(fc["q95"]), "skill": array(fc["skill"])}
        if payload.get("fit_summary"):
            model.fit_summary_ = FitSummary.from_dict(payload["fit_summary"])
        tail = payload.get("tail")
        if tail and tail.get("values") is not None:
            model._tail = TelemetryTable(
                sensors=list(model.sensors), values=array(tail["values"]), controls=list(model.controls),
                control_values=array(tail["controls"]), timestamps=tail.get("timestamps"),
                time_seconds=array(tail.get("time_seconds")), time_column=model.time_column,
                step_values=array(tail.get("step_values")), time_offset=tail.get("time_offset"),
            )
            model._tail_offset = int(tail.get("offset", 0))
        if payload.get("companion"):
            model._free = cls._from_payload(payload["companion"], device)
        return model


class StreamMonitor:
    """
    Score live telemetry one row at a time.

        monitor = model.stream()
        for row in incoming_rows:                 # dict of {column: value}
            step = monitor.update(row)
            if step.alert:
                print(step.time, step.sensors[0]["name"], step.score)

    With `adapt=True` the monitor works on its own copy of the engine and keeps learning
    from the stream (the engine's online plasticity), so a lasting change is flagged when it
    starts and then accepted as the new normal.
    """

    def __init__(self, model: TelemetryModel, adapt: bool = False, sensitivity: Union[str, float] = "medium"):
        model._require_fitted()
        self.model = model
        self.adapt = bool(adapt)
        self.threshold = resolve_threshold(sensitivity)
        self.engine = copy.deepcopy(model.engine) if self.adapt else model.engine
        self.engine.eval()
        if self.adapt:
            self.engine.reset_runtime_adaptation()
        self.index = 0
        n, m = len(model.sensors), len(model.controls)
        self._field: Optional[torch.Tensor] = None
        self._prev_x: Optional[torch.Tensor] = None
        self._prev_q: Optional[np.ndarray] = None        # last real (not missing) normalised level of every sensor
        self._gap = np.zeros(n)                          # missing readings since then
        self._prev_un = np.zeros(m)
        self._raw_ctrl: Optional[np.ndarray] = None
        self._score_state: Optional[Dict[str, np.ndarray]] = None
        self._streak = 0

    @torch.no_grad()
    def _advance(self, state: np.ndarray, action: Optional[np.ndarray]) -> np.ndarray:
        """
        Consume one row. `state` is what the engine sees of it; `action` is the control input
        for the transition from the previous row to this one (None for the very first row).
        Returns the velocity that the engine predicted for this row before seeing it.
        """
        model, n = self.model, len(self.model.sensors)
        x = torch.as_tensor(state, dtype=torch.float32, device=model._device).view(1, 1, -1)
        predicted = np.zeros(n)
        if self._prev_x is not None and action is not None:
            a = torch.as_tensor(action, dtype=torch.float32, device=model._device).view(1, 1, -1)
            out = self.engine(self._prev_x, a, standing_field=self._field, plastic_runtime=False)
            predicted = out["output"][0, 0, n:2 * n].double().cpu().numpy()
            if self.adapt:
                # The true state is known now, so let the engine learn from the transition it has
                # just been scored on. This happens *after* its prediction was recorded: no leakage.
                self.engine(self._prev_x, a, standing_field=self._field, plastic_runtime=True, y_target=x)
            self._field = out["standing_field"]
        self._prev_x = x
        return predicted

    def update(self, row: Any, controls: Any = None, timestamp: Optional[str] = None) -> StepResult:
        """
        Feed the next reading. `row` is a dict {column: value} (extra keys are ignored) or a
        sequence of sensor values in the model's sensor order. A value that is missing, or
        not a finite number, is treated as "no reading": the last real value is held and that
        sensor is not scored for this row.
        """
        model = self.model
        n, m = len(model.sensors), len(model.controls)
        if isinstance(row, dict):
            def number(value: Any) -> float:
                return math.nan if value is None or value == "" else float(value)

            try:
                values = np.array([number(row[name]) for name in model.sensors], dtype=float)
            except KeyError as exc:
                raise DataError(f"Reading is missing sensor {exc}.") from exc
            except (TypeError, ValueError) as exc:
                raise DataError(f"Reading contains a non-numeric value: {exc}.") from exc
            try:
                ctrl = np.array([number(row.get(name)) for name in model.controls], dtype=float)
            except (TypeError, ValueError) as exc:
                raise DataError(f"Reading contains a non-numeric control value: {exc}.") from exc
            if timestamp is None:
                for key in ([model.time_column] if model.time_column else []) + ["timestamp", "time", "datetime", "date", "ts"]:
                    if row.get(key) not in (None, ""):
                        timestamp = str(row[key])
                        break
        else:
            try:
                values = np.asarray(row, dtype=float).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise DataError(f"Reading contains a non-numeric value: {exc}.") from exc
            if values.shape[0] != n:
                raise DataError(f"Expected {n} sensor values, got {values.shape[0]}.")
            if controls is None:
                ctrl = np.full(m, math.nan)
            else:
                try:
                    ctrl = np.asarray(controls, dtype=float).reshape(-1)
                except (TypeError, ValueError) as exc:
                    raise DataError(f"Reading contains a non-numeric control value: {exc}.") from exc
                if ctrl.shape[0] != m:
                    raise DataError(f"Expected {m} control value(s) ({', '.join(model.controls) or 'none'}), got {ctrl.shape[0]}.")
        values = np.where(np.isfinite(values), values, np.nan)
        ctrl = np.where(np.isfinite(ctrl), ctrl, np.nan)
        missing = np.isnan(values)
        first = self._prev_q is None
        if first and missing.any():
            raise DataError("The first reading of a stream cannot contain missing values.")
        if m:
            # A control that is not reported stays where it was (its typical value, at the very start).
            ctrl = np.where(np.isnan(ctrl), model._cmean if self._raw_ctrl is None else self._raw_ctrl, ctrl)

        qn_now = (values - model._mean) / model._std
        qn = qn_now if first else np.where(missing, self._prev_q, qn_now)
        gap = np.where(missing, 0.0, self._gap)
        previous = qn if first else self._prev_q
        dq = qn - previous
        p = np.zeros(n) if first else (dq - (gap + 1.0) * model._vmean) / model._vs
        if m:
            un = (ctrl - model._cmean) / model._cstd
            du = np.zeros(m) if first else (un - self._prev_un) / model._cds
            ahead = model._shift > 0
            action = np.concatenate([np.clip(np.where(ahead, un, self._prev_un), -_CLIP, _CLIP),
                                     np.clip(np.where(ahead, du, 0.0), -_CLIP, _CLIP)])
        else:
            un, du, action = np.zeros(0), np.zeros(0), np.zeros(1)
        p_hat = self._advance(model._states(qn, p), None if first else action)

        valid = ~missing
        residual = np.clip((p - model._gain * p_hat) / np.sqrt(gap + 1.0), -_Z_CAP, _Z_CAP)[None, :]
        if self.index < 2:
            residual[:] = 0.0                                 # as in `detect`: the first two rows cannot surprise
        moved = np.full(n, np.nan) if first else dq           # no previous reading: not "unchanged" either
        scored = score_residuals(residual, level=qn[None, :], delta=moved[None, :], cal=model._cal,
                                 state=self._score_state, valid=valid[None, :], range_check=not self.adapt)
        self._score_state = scored.state
        sensor_score = np.minimum(scored.sensor_score[0], _Z_CAP)
        score = float(min(scored.score[0], _Z_CAP))
        expected_qn = qn if first else previous + (gap + 1.0) * model._vmean + model._vs * model._gain * p_hat
        expected = expected_qn * model._std + model._mean
        shown = qn * model._std + model._mean                # the reading, or the held value where there was none

        flagged = []
        for i in np.argsort(-sensor_score):
            if sensor_score[i] <= self.threshold:
                break
            reason = REASONS[int(scored.reason[0, i])]
            signed = scored.z[0, i] if reason == "surprise" else scored.drift_z[0, i]
            direction = "above" if signed >= 0 else "below"
            if reason == "flatline":
                detail = f"reading stuck at {_fmt(shown[i])}"
            elif reason == "range":
                direction = "above" if shown[i] >= model._mean[i] else "below"
                detail = f"{direction} the range seen in training ({_fmt(shown[i])})"
            else:
                detail = f"{REASON_TEXT[reason]} {direction} expected ({' vs '.join(_fmt_pair(shown[i], expected[i]))})"
            flagged.append({"name": model.sensors[i], "score": round(float(sensor_score[i]), 3),
                            "reason": REASON_TEXT[reason], "direction": direction, "detail": detail})
        moved = [model.controls[j] for j in np.flatnonzero(np.abs(du) > model._cnote)] if m and not first else []

        # ---- remember this row --------------------------------------------------------------
        self._prev_q = qn
        self._gap = np.where(missing, self._gap + 1.0, 0.0)
        if m:
            self._prev_un, self._raw_ctrl = un, ctrl
        is_anomaly = score > self.threshold
        self._streak = self._streak + 1 if is_anomaly else 0
        result = StepResult(
            index=self.index, time=timestamp or f"step {self.index}", score=score, is_anomaly=is_anomaly,
            alert=is_anomaly and (self._streak >= 2 or score >= 1.25 * self.threshold),
            sensors=flagged,
            expected={s: float(v) for s, v in zip(model.sensors, expected)},
            values={s: float(v) for s, v in zip(model.sensors, shown)},
            sensor_scores={s: float(v) for s, v in zip(model.sensors, sensor_score)},
            controls_changed=moved,
        )
        self.index += 1
        return result
