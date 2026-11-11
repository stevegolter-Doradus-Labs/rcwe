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
trained to predict the next state from the current state and the control inputs. Three
things keep the results honest:

* No target leakage. Predictions are always made *before* the true next value is shown to
  the engine (the engine's posterior world-line weighting is only used for the optional
  online-adaptation update, after scoring).
* A naive baseline is always in the room. Accuracy is reported as skill versus "next value
  = current value", and a per-sensor calibration shrinks the engine's prediction toward
  that baseline wherever the engine does not beat it on held-out data.
* Calibration happens on data the engine was not trained on (the last part of the log).
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.kernels.accelerator import describe_accelerator, select_best_device
from rcwe.toolkit.results import AnomalyEvent, DetectionResult, FitSummary, Forecast, _fmt
from rcwe.toolkit.scoring import (
    REASON_TEXT, REASONS, ScoreCalibration, detection_metrics, ewma, find_events,
    longest_run, resolve_threshold, robust_sigma, score_residuals,
)
from rcwe.toolkit.table import DataError, TelemetryTable, format_time, load_table

__all__ = ["TelemetryModel", "StreamMonitor", "StepResult", "NotFittedError"]

FORMAT_NAME = "rcwe-telemetry-model"
FORMAT_VERSION = 1
_CLIP = 8.0            # model inputs are clipped to +/- this many (normalised) units
_Z_CAP = 1.0e4         # scores are capped here so that broken sensors do not print as 1e17
_CHUNK = 8192          # rows per forward pass when scoring long logs
_CONTEXT = 128         # rows of history used to warm the standing wavefield before forecasting
ProgressFn = Callable[[str, float, str], None]


class NotFittedError(RuntimeError):
    """Raised when a model is used before `fit()` or `load()`."""


def _stable_envelope(qn: np.ndarray, n_train: int) -> np.ndarray:
    """
    Per sensor: does the signal stay inside a stable operating envelope?

    False for signals with a strong trend (counters, totalisers), for random-walk-like signals
    whose range keeps growing (each quarter of the training log spans much less than the whole),
    and for signals whose held-out tail already left the envelope of the training part.
    Logs that are too short to judge are given the benefit of the doubt.
    """
    T, n = qn.shape
    stable = np.ones(n, dtype=bool)
    if n_train < 80:
        return stable
    train = qn[:n_train]
    width = np.ptp(train, axis=0)
    steps = np.arange(n_train, dtype=float)
    for i in range(n):
        if width[i] <= 0:
            continue
        with np.errstate(invalid="ignore"):
            trend = abs(float(np.corrcoef(steps, train[:, i])[0, 1]))
        blocks = np.array([np.ptp(b) for b in np.array_split(train[:, i], 4)])
        if (np.isfinite(trend) and trend >= 0.9) or width[i] > 1.4 * float(np.median(blocks)):
            stable[i] = False
            continue
        if n_train < T:
            margin = max(0.25 * width[i], 0.25)
            tail = qn[n_train:, i]
            if tail.max() > train[:, i].max() + margin or tail.min() < train[:, i].min() - margin:
                stable[i] = False
    return stable


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

    __slots__ = ("index", "time", "score", "is_anomaly", "alert", "sensors", "expected", "values", "sensor_scores")

    def __init__(self, index: int, time: str, score: float, is_anomaly: bool, sensors: List[Dict[str, Any]],
                 expected: Dict[str, float], values: Dict[str, float], sensor_scores: Dict[str, float], alert: bool = False):
        self.index = index
        self.time = time
        self.score = score
        self.is_anomaly = is_anomaly      # this reading is above the threshold
        self.alert = alert                # ...and it is not a one-off blip (two in a row, or clearly above)
        self.sensors = sensors            # sensors above threshold, worst first: {name, score, reason, direction, detail}
        self.expected = expected
        self.values = values
        self.sensor_scores = sensor_scores

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "time": self.time, "score": round(self.score, 4), "is_anomaly": self.is_anomaly,
            "alert": self.alert, "sensors": self.sensors, "expected": self.expected, "values": self.values,
            "sensor_scores": {k: round(v, 4) for k, v in self.sensor_scores.items()},
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
        self.time_column: Optional[str] = None
        self.fit_summary_: Optional[FitSummary] = None
        self._device = torch.device("cpu")
        self._mean = self._std = self._vs = self._vmean = self._cmean = self._cstd = self._gain = None
        self._level_mask: Optional[np.ndarray] = None   # 0 hides a non-stationary sensor's level from the engine
        self._cal: Optional[ScoreCalibration] = None
        self._fc: Optional[Dict[str, np.ndarray]] = None
        self._tail: Optional[Dict[str, Any]] = None
        self._fit_table: Optional[TelemetryTable] = None

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
        return torch.device(choice)

    # ------------------------------------------------------------------------------
    # data plumbing
    # ------------------------------------------------------------------------------
    def _aligned(self, table: TelemetryTable) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Pick this model's sensor/control columns out of `table` (by name)."""
        notes: List[str] = []
        positional = (
            table.n_sensors == len(self.sensors)
            and all(name.startswith("sensor_") for name in table.sensors)
            and not set(self.sensors) & set(table.sensors)
        )
        cols = []
        for i, name in enumerate(self.sensors):
            if name in table.sensors:
                cols.append(table.values[:, table.sensors.index(name)])
            elif name in table.controls:
                cols.append(table.control_values[:, table.controls.index(name)])
            elif positional:
                cols.append(table.values[:, i])
            else:
                have = ", ".join(table.sensors[:10]) + ("..." if table.n_sensors > 10 else "")
                raise DataError(
                    f"Sensor '{name}' that this model was trained on is missing from the data. "
                    f"Columns found: {have}."
                )
        values = np.stack(cols, axis=1)
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
        controls = np.stack(ctrl_cols, axis=1) if ctrl_cols else np.zeros((table.n_rows, 0))
        return values, controls, notes

    def _normalise(self, values: np.ndarray, controls: np.ndarray,
                   previous: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return level-normalised q, its step change dq, velocity p, and normalised controls."""
        qn = (values - self._mean) / self._std
        first = qn[:1] if previous is None else previous.reshape(1, -1)
        dq = np.diff(qn, axis=0, prepend=first)
        p = (dq - self._vmean) / self._vs          # standardised velocity: 0 means "the usual change"
        if previous is None:
            p[0] = 0.0                              # no previous row: assume the usual change
        if controls.shape[1]:
            un = (np.where(np.isnan(controls), self._cmean, controls) - self._cmean) / self._cstd
        else:
            un = np.zeros((values.shape[0], 1))
        return qn, dq, p, un

    def _tensors(self, qn: np.ndarray, p: np.ndarray, un: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        state = np.concatenate([np.clip(qn, -_CLIP, _CLIP) * self._level_mask, np.clip(p, -_CLIP, _CLIP)], axis=1)
        x = torch.as_tensor(state, dtype=torch.float32, device=self._device)
        a = torch.as_tensor(np.clip(un, -_CLIP, _CLIP), dtype=torch.float32, device=self._device)
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

    # ------------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------------
    def fit(
        self,
        data: Any,
        *,
        sensors: Optional[Sequence[str]] = None,
        controls: Optional[Sequence[str]] = None,
        time_column: Optional[str] = None,
        label_column: Optional[str] = None,
        progress: Optional[ProgressFn] = None,
    ) -> "TelemetryModel":
        """
        Learn normal behaviour from a telemetry log. Returns `self`, so calls can be chained.

        `data` may be a file path (CSV/TSV/JSON/JSONL), a list of row dicts, a dict of columns,
        a NumPy array, a pandas DataFrame or a `TelemetryTable`. Train on a period of normal
        operation when you can; a modest share of unlabelled anomalies in the log is tolerated
        because calibration uses robust statistics. Label columns are never used for training.
        """
        table = load_table(data, sensors=sensors, controls=controls, time_column=time_column, label_column=label_column)
        notes = list(table.notes)
        warnings: List[str] = []
        T, n = table.n_rows, table.n_sensors
        if T < 12:
            raise DataError(f"Training needs at least 12 time steps; this log has {T}.")
        say = progress or (lambda stage, fraction, message: None)
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

        self.sensors, self.controls = list(table.sensors), list(table.controls)
        self.time_column = table.time_column
        values, ctrl = table.values, table.control_values

        # ---- scalers (from the training part only) ------------------------------------
        self._mean = values[:n_tr].mean(axis=0)
        std = values[:n_tr].std(axis=0)
        constant = std < 1e-12 * np.maximum(np.abs(self._mean), 1.0)
        self._std = np.where(constant, 1.0, std)
        qn_all = (values - self._mean) / self._std
        dq_all = np.diff(qn_all, axis=0, prepend=qn_all[:1])
        vs = dq_all[1:n_tr].std(axis=0)
        self._vs = np.where(vs < 1e-9, 1.0, vs)
        self._vmean = dq_all[1:n_tr].mean(axis=0)   # the usual per-step change (non-zero for counters and trends)
        if ctrl.shape[1]:
            self._cmean = ctrl[:n_tr].mean(axis=0)
            cstd = ctrl[:n_tr].std(axis=0)
            self._cstd = np.where(cstd < 1e-12, 1.0, cstd)
        else:
            self._cmean, self._cstd = np.zeros(0), np.ones(0)
        self._gain = np.ones(n)
        if constant.any():
            notes.append("Sensor(s) constant during training (any change will be flagged): "
                         + ", ".join(s for s, c in zip(self.sensors, constant) if c) + ".")

        # ---- which sensors have a stable operating envelope? ------------------------------
        # Counters, totalisers and random-walk-like signals keep leaving the range they were
        # trained on. For those, only the rate of change is modelled: their absolute level is
        # hidden from the engine (so it cannot extrapolate) and the range check is switched off.
        stationary = _stable_envelope(qn_all, n_tr)
        self._level_mask = stationary.astype(float)
        free = [s for s, ok in zip(self.sensors, stationary) if not ok]
        if free:
            notes.append("Free-running sensor(s) (trend or random walk): only their rate of change is modelled, "
                         "and the range check is off: " + ", ".join(free) + ".")

        qn, dq, p, un = self._normalise(values, ctrl)
        m = un.shape[1]

        # ---- build the engine ------------------------------------------------------------
        d_model = cfg["d_model"]
        if d_model == "auto":
            d_model = 64 if n <= 24 else int(min(256, 32 * math.ceil(2.5 * n / 32)))
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(cfg["seed"])
        try:
            device = self._resolve_device()
            try:
                history = self._train(device, d_model, m, qn, p, un, n_tr, tiny, say)
            except RuntimeError as exc:
                if device.type == "cpu":
                    raise
                warnings.append(f"Training on {device.type} failed ({str(exc).splitlines()[0][:120]}); used the CPU instead.")
                torch.manual_seed(cfg["seed"])
                history = self._train(torch.device("cpu"), d_model, m, qn, p, un, n_tr, tiny, say)
        finally:
            torch.random.set_rng_state(rng_state)

        # ---- calibrate on held-out data ------------------------------------------------
        say("calibrate", 0.9, "Calibrating scores and forecast intervals")
        x, a = self._tensors(qn, p, un)
        p_hat_next, _ = self._predict_frozen(x[:-1], a[:-1])          # prediction of p at rows 1..T-1
        target = p[1:]
        cal_slice = slice(0, T - 1) if n_val == 0 else slice(n_tr - 1, T - 1)
        ph, tg = p_hat_next[cal_slice], target[cal_slice]
        num, den = (ph * tg).sum(axis=0), (ph * ph).sum(axis=0)
        gain = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
        self._gain = np.clip(gain, 0.0, 1.0 if n_val == 0 else 1.5)
        resid = tg - self._gain * ph
        base = (tg ** 2).mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sensor_skill = np.where(base > 1e-12, 1.0 - (resid ** 2).mean(axis=0) / base, np.nan)
        skill = float(np.nanmean(sensor_skill)) if np.isfinite(sensor_skill).any() else float("nan")

        lam = 0.25
        # Residual scale. The floor says "no model is credibly more than 20x (3x for a log too
        # short to hold data back) better than the naive baseline", which keeps an over-fitted
        # model from reporting absurd sigma counts.
        floor = 0.3 if n_val == 0 else 0.05
        sigma = robust_sigma(resid, floor=floor)
        ewma_sigma = np.maximum(robust_sigma(ewma(resid, lam), floor=floor * math.sqrt(lam / (2 - lam))),
                                sigma * math.sqrt(lam / (2 - lam)))
        if n_val == 0:
            sigma, ewma_sigma = sigma * 1.5, ewma_sigma * 1.5

        lo, hi = qn.min(axis=0), qn.max(axis=0)
        width = hi - lo
        margin = np.where(width > 0, np.maximum(0.25 * width, 0.25),
                          np.maximum(1e-9, 1e-6 * np.abs(self._mean)) / self._std)
        # An envelope learned from a short log is too narrow to be trusted as a limit.
        range_enabled = stationary.astype(float) * (1.0 if n_tr >= 200 else 0.0)

        abs_dq = np.abs(dq[1:])
        flat_eps = np.zeros(n)
        flat_run = np.zeros(n)
        for i in range(n):
            moving = abs_dq[:, i][abs_dq[:, i] > 0]
            if moving.size == 0:
                continue                                   # constant sensor: the range check covers it
            flat_eps[i] = 1e-3 * float(np.median(moving))
            longest = longest_run(abs_dq[:, i] <= flat_eps[i])
            if longest <= (T - 1) / 5:
                flat_run[i] = max(8, math.ceil(2.5 * longest) + 2)

        self._cal = ScoreCalibration(sigma=sigma, ewma_sigma=ewma_sigma, ewma_lambda=lam, range_lo=lo, range_hi=hi,
                                     range_margin=margin, range_enabled=range_enabled, flat_eps=flat_eps, flat_run=flat_run)

        # ---- backtest multi-step forecasts ----------------------------------------------
        cal_start = 0 if n_val == 0 else n_tr
        self._fc = self._calibrate_forecasts(qn, p, un, cal_start, T)
        forecast_skill: Dict[str, float] = {}
        horizon = int(self._fc["weights"].shape[0]) if self._fc["backtested"] else 0
        if horizon:
            overall = np.nanmean(self._fc["skill"], axis=1)
            for h in (1, 6, 12, 24, 48, 96):
                if h <= horizon and np.isfinite(overall[h - 1]):
                    forecast_skill[str(h)] = float(overall[h - 1])

        # ---- remember the end of the log so forecasts can continue from it ------------
        keep = min(T, 3 * _CONTEXT)
        self._tail = {
            "values": values[-keep:].copy(),
            "controls": ctrl[-keep:].copy(),
            "timestamps": None if table.timestamps is None else list(table.timestamps[-keep:]),
            "time_seconds": None if table.time_seconds is None else table.time_seconds[-keep:].copy(),
            "sample_interval": table.sample_interval,
            "rows": T,
        }
        self._fit_table = table

        weak = [s for s, k in zip(self.sensors, sensor_skill) if np.isfinite(k) and k < 0.02]
        if np.isfinite(skill) and skill < 0.02:
            warnings.append("The model predicts the next reading no better than repeating the current one. "
                            "Anomaly scoring still works, but expect little from forecasts.")
        elif weak and len(weak) < n:
            notes.append("No predictive skill beyond 'last value' for: " + ", ".join(weak) + ".")

        self.fit_summary_ = FitSummary(
            rows=T, train_rows=n_tr, validation_rows=n_val, sensors=list(self.sensors), controls=list(self.controls),
            steps_run=history["steps"], best_step=history["best_step"], seconds=history["seconds"],
            device=describe_accelerator(self._device),
            parameters=sum(prm.numel() for prm in self.engine.parameters()),
            skill=skill, sensor_skill={s: float(k) for s, k in zip(self.sensors, sensor_skill)},
            forecast_skill=forecast_skill, forecast_horizon=horizon,
            warnings=warnings, notes=notes, source=table.source,
        )
        say("done", 1.0, "Model ready")
        return self

    def _train(self, device: torch.device, d_model: int, m: int, qn: np.ndarray, p: np.ndarray, un: np.ndarray,
               n_tr: int, tiny: bool, say: ProgressFn) -> Dict[str, Any]:
        cfg = self.settings
        n = qn.shape[1]
        self._device = device
        engine = ResonantCausalWorldEngine(
            d_input=2 * n, d_model=int(d_model), d_action=m,
            num_worldlines=cfg["num_worldlines"], num_freq_bands=cfg["num_freq_bands"],
        ).to(device)
        self.engine = engine
        x, a = self._tensors(qn, p, un)
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
    def _rollout(self, q0: np.ndarray, p0: np.ndarray, field: Optional[torch.Tensor], future_u: np.ndarray) -> np.ndarray:
        """Free-run the engine `H` steps from states (q0, p0) [B, n]; returns level-normalised q-hat [B, H, n]."""
        n = len(self.sensors)
        dev = self._device
        q = torch.as_tensor(q0, dtype=torch.float32, device=dev)
        p = torch.as_tensor(p0, dtype=torch.float32, device=dev)
        u = torch.as_tensor(np.clip(future_u, -_CLIP, _CLIP), dtype=torch.float32, device=dev)
        vs = torch.as_tensor(self._vs, dtype=torch.float32, device=dev)
        vmean = torch.as_tensor(self._vmean, dtype=torch.float32, device=dev)
        gain = torch.as_tensor(self._gain, dtype=torch.float32, device=dev)
        mask = torch.as_tensor(self._level_mask, dtype=torch.float32, device=dev)
        steps = []
        for h in range(u.shape[1]):
            x = torch.cat([q.clamp(-_CLIP, _CLIP) * mask, p.clamp(-_CLIP, _CLIP)], dim=-1).unsqueeze(1)
            out = self.engine(x, u[:, h:h + 1, :], standing_field=field, plastic_runtime=False)
            field = out["standing_field"]
            p = out["output"][:, 0, n:2 * n] * gain
            q = q + vmean + vs * p
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

    def _calibrate_forecasts(self, qn: np.ndarray, p: np.ndarray, un: np.ndarray, lo: int, hi: int) -> Dict[str, Any]:
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
        x, a = self._tensors(qn, p, un)
        field = self._warm_field(x, a, anchors)
        held = np.repeat(un[anchors][:, None, :], H, axis=1)   # controls unknown in advance: hold the last value
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
            table = load_table(data)
            extra_notes = []
        if table.n_rows < 3:
            raise DataError("Detection needs at least 3 time steps.")
        threshold = resolve_threshold(sensitivity)
        use_range = (not adapt) if range_check is None else bool(range_check)
        values, ctrl, align_notes = self._aligned(table)
        qn, dq, p, un = self._normalise(values, ctrl)
        T, n = qn.shape
        say = progress or (lambda stage, fraction, message: None)

        if adapt:
            monitor = StreamMonitor(self, adapt=True, sensitivity=threshold)
            p_hat = np.zeros((T, n))
            for t in range(T):
                p_hat[t] = monitor._advance(qn[t], p[t], un[t])
                if t % 200 == 0:
                    say("detect", t / T, f"Replaying with online adaptation: row {t:,}/{T:,}")
        else:
            x, a = self._tensors(qn, p, un)
            p_next, _ = self._predict_frozen(x[:-1], a[:-1])
            p_hat = np.vstack([np.zeros((1, n)), p_next])

        valid = np.ones(T, dtype=bool)
        valid[0] = False                                   # the first row has nothing to be compared with
        residual = np.clip(p - self._gain * p_hat, -_Z_CAP, _Z_CAP)
        scored = score_residuals(residual, level=qn, delta=dq, cal=self._cal, valid=valid, range_check=use_range)
        sensor_scores = np.minimum(scored.sensor_score, _Z_CAP)
        scores = np.minimum(scored.score, _Z_CAP)

        expected_qn = qn.copy()
        expected_qn[1:] = qn[:-1] + self._vmean + self._vs * self._gain * p_hat[1:]
        expected = expected_qn * self._std + self._mean
        band = threshold * self._cal.sigma * self._vs * self._std

        stamps = [table.time_label(i) for i in range(T)]
        spans = find_events(scores, threshold, merge_gap=merge_gap, min_duration=min_duration)
        events = [self._describe_event(k, a_, b_, scored, sensor_scores, scores, threshold, values, expected, stamps)
                  for k, (a_, b_) in enumerate(spans, 1)]

        metrics = None
        if table.labels is not None and table.labels.shape[0] == T:
            metrics = detection_metrics(table.labels, scores, spans, tolerance=merge_gap)

        notes = extra_notes + align_notes
        gaps = int(table.missing.sum())
        if gaps:
            notes.append(f"{gaps} missing reading(s) were filled by interpolation before scoring.")
        if adapt:
            notes.append("Adaptive mode: the model kept learning during the replay, so events mark the onset of "
                         "changes rather than their full duration.")
        say("detect", 1.0, f"{len(events)} event(s) found")
        return DetectionResult(
            sensors=list(self.sensors), timestamps=stamps, values=values, expected=expected, band=band,
            residual_z=np.clip(scored.z, -_Z_CAP, _Z_CAP), sensor_scores=sensor_scores, scores=scores,
            threshold=threshold, events=events, mode="adaptive" if adapt else "reference",
            time_seconds=table.time_seconds, labels=table.labels, metrics=metrics, notes=notes,
            source=table.source, model_info=self._model_info(),
        )

    def _describe_event(self, index: int, a: int, b: int, scored: Any, sensor_scores: np.ndarray, scores: np.ndarray,
                        threshold: float, values: np.ndarray, expected: np.ndarray, stamps: List[str]) -> AnomalyEvent:
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
            if reason == "range":
                t_i = a + int(np.argmax(scored.range_z[a:b + 1, i]))
                signed = float(values[t_i, i] - self._mean[i])
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

        when = stamps[a] if a == b else f"{stamps[a]} to {stamps[b]}"
        length = "1 step" if a == b else f"{b - a + 1} steps"
        if involved:
            lead = involved[0]
            if lead["reason"] == "flatline":
                what = f"{lead['name']} stopped changing (stuck at {_fmt(lead['actual'])})"
            elif lead["reason"] == "range":
                what = (f"{lead['name']} went {lead['direction']} the range seen in training "
                        f"(reached {_fmt(lead['actual'])})")
            elif kind == "spike":
                what = (f"{lead['name']} jumped {lead['direction']} expectations "
                        f"({_fmt(lead['actual'])} vs {_fmt(lead['expected'])} expected)")
            else:
                what = (f"{lead['name']} ran {lead['direction']} expectations "
                        f"({_fmt(lead['actual'])} vs {_fmt(lead['expected'])} expected at the peak)")
            others = [s["name"] for s in involved[1:4]]
            also = f"; also involved: {', '.join(others)}" if others else ""
            finding = f"{what}, peaking at {peak_score:.1f} sigma{also}."
        else:
            finding = f"anomaly score peaked at {peak_score:.1f} sigma."
        description = f"{when} ({length}): {finding}"
        return AnomalyEvent(
            index=index, start=a, end=b, start_time=stamps[a], end_time=stamps[b], peak=peak, peak_time=stamps[peak],
            peak_score=peak_score, severity=_severity(peak_score / threshold), kind=kind,
            sensors=involved, description=description, what=finding,
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
            it, controls are assumed to stay at their latest values.
        """
        self._require_fitted()
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        notes: List[str] = []
        if data is None:
            tail = self._tail
            if tail is None:
                raise DataError("No data given and the model does not remember the end of its training log.")
            values, ctrl = tail["values"], tail["controls"]
            stamps, seconds, interval, row_index = tail["timestamps"], tail["time_seconds"], tail["sample_interval"], tail["rows"]
        else:
            table = load_table(data)
            if table.n_rows < 2:
                raise DataError("Forecasting needs at least 2 rows of recent history.")
            values, ctrl, align_notes = self._aligned(table)
            notes.extend(align_notes)
            stamps, seconds, interval, row_index = table.timestamps, table.time_seconds, table.sample_interval, table.n_rows
            if interval is None and self._tail is not None:
                interval = self._tail["sample_interval"]

        qn, dq, p, un = self._normalise(values[-_CONTEXT:], ctrl[-_CONTEXT:])
        x, a = self._tensors(qn, p, un)
        field = None
        if x.shape[0] > 1:
            with torch.no_grad():
                field = self.engine(x[:-1].unsqueeze(0), a[:-1].unsqueeze(0), plastic_runtime=False)["standing_field"]

        future = np.repeat(un[-1][None, None, :], steps, axis=1)
        if future_controls is not None and self.controls:
            planned = future_controls
            if not isinstance(planned, np.ndarray):
                try:
                    planned_table = load_table(planned)
                    planned = np.stack([planned_table.column(c) for c in self.controls], axis=1)
                except (DataError, KeyError) as exc:
                    raise DataError(f"future_controls could not be read: {exc}") from exc
            planned = np.asarray(planned, dtype=float).reshape(-1, len(self.controls))
            if planned.shape[0] < steps:
                notes.append(f"future_controls covers {planned.shape[0]} of {steps} steps; the last row is held after that.")
                planned = np.vstack([planned, np.repeat(planned[-1:], steps - planned.shape[0], axis=0)])
            future = ((planned[:steps] - self._cmean) / self._cstd)[None, :, :]
        elif self.controls:
            notes.append("Controls were assumed to stay at their latest values.")

        rolled = self._rollout(qn[-1:], p[-1:], field, future)[0]            # [steps, n]
        fc = self._fc
        H = fc["weights"].shape[0]
        idx = np.minimum(np.arange(steps), H - 1)
        growth = np.sqrt(np.maximum(np.arange(1, steps + 1) / H, 1.0))[:, None]
        weights = fc["weights"][idx]
        naive = qn[-1][None, :] + np.arange(1, steps + 1)[:, None] * self._vmean[None, :]
        centre = naive + weights * (rolled - naive)
        q80, q95 = fc["q80"][idx] * growth, fc["q95"][idx] * growth
        if not fc["backtested"]:
            spread = np.sqrt(np.arange(1, steps + 1))[:, None]
            q80, q95 = fc["q80"][idx] * spread, fc["q95"][idx] * spread
            notes.append("Too little held-out data to backtest forecasts; intervals are rough estimates.")
        elif steps > H:
            notes.append(f"Horizons beyond {H} steps were not backtested; intervals there are extrapolated.")

        def unscale(arr: np.ndarray) -> np.ndarray:
            return arr * self._std + self._mean

        labels: List[str] = []
        future_seconds: Optional[np.ndarray] = None
        if seconds is not None and interval:
            future_seconds = float(seconds[-1]) + interval * np.arange(1, steps + 1)
            labels = [format_time(s) for s in future_seconds]
        else:
            labels = [f"step {row_index + h}" for h in range(steps)]
        hist = min(values.shape[0], max(3 * steps, 48), 3 * _CONTEXT)
        skill_row = fc["skill"][min(steps, H) - 1]
        return Forecast(
            sensors=list(self.sensors), values=unscale(centre), lower=unscale(centre - q80), upper=unscale(centre + q80),
            lower95=unscale(centre - q95), upper95=unscale(centre + q95), timestamps=labels, time_seconds=future_seconds,
            origin=str(stamps[-1]) if stamps is not None else f"step {row_index - 1}",
            history=values[-hist:].copy(),
            history_timestamps=None if stamps is None else [str(s) for s in stamps[-hist:]],
            sensor_skill={s: float(k) for s, k in zip(self.sensors, skill_row)},
            model_weight={s: float(w) for s, w in zip(self.sensors, weights.mean(axis=0))},
            notes=notes,
        )

    # ------------------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> str:
        """Save everything needed to reuse the model (weights, scalers, calibration) in one file."""
        self._require_fitted()

        def tensor(arr: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            return None if arr is None else torch.as_tensor(np.asarray(arr), dtype=torch.float64)

        from rcwe import __version__
        tail = self._tail or {}
        payload = {
            "format": FORMAT_NAME, "format_version": FORMAT_VERSION, "rcwe_version": __version__,
            "settings": dict(self.settings),
            "engine_config": {
                "d_input": self.engine.d_input, "d_model": self.engine.d_model, "d_action": self.engine.d_action,
                "num_worldlines": self.engine.num_worldlines, "num_freq_bands": self.engine.num_freq_bands,
            },
            "engine_state": {k: v.detach().cpu() for k, v in self.engine.state_dict().items()},
            "sensors": list(self.sensors), "controls": list(self.controls), "time_column": self.time_column,
            "scaler": {"mean": tensor(self._mean), "std": tensor(self._std), "vs": tensor(self._vs),
                       "vmean": tensor(self._vmean),
                       "cmean": tensor(self._cmean), "cstd": tensor(self._cstd), "gain": tensor(self._gain),
                       "level_mask": tensor(self._level_mask)},
            "calibration": self._cal.to_payload(),
            "forecast": {"backtested": bool(self._fc["backtested"]), "weights": tensor(self._fc["weights"]),
                         "q80": tensor(self._fc["q80"]), "q95": tensor(self._fc["q95"]), "skill": tensor(self._fc["skill"])},
            "fit_summary": None if self.fit_summary_ is None else self.fit_summary_.to_dict(),
            "tail": {
                "values": tensor(tail.get("values")), "controls": tensor(tail.get("controls")),
                "timestamps": tail.get("timestamps"), "time_seconds": tensor(tail.get("time_seconds")),
                "sample_interval": tail.get("sample_interval"), "rows": int(tail.get("rows", 0)),
            },
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)
        return str(out.resolve())

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "auto") -> "TelemetryModel":
        """Load a model written by `save()`. Uses PyTorch's safe `weights_only` loader."""
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"Model file not found: {file}")
        try:
            payload = torch.load(file, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - surface a readable message for any unpickling problem
            raise DataError(f"{file.name} is not a readable RCWE model file ({str(exc).splitlines()[0][:160]}).") from exc
        if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
            hint = ""
            if isinstance(payload, dict) and "state_dict" in payload and "config" in payload:
                hint = " It looks like a raw engine checkpoint: open it with ResonantCausalWorldEngine.load_pretrained()."
            raise DataError(f"{file.name} is not an RCWE telemetry model.{hint}")
        if int(payload.get("format_version", 0)) > FORMAT_VERSION:
            raise DataError(f"{file.name} was written by a newer RCWE version (format {payload['format_version']}); please upgrade.")

        settings = dict(payload["settings"])
        settings["device"] = device
        model = cls(**settings)
        model._device = model._resolve_device()
        cfg = payload["engine_config"]
        engine = ResonantCausalWorldEngine(**{k: int(v) for k, v in cfg.items()})
        engine.load_state_dict(payload["engine_state"])
        model.engine = engine.to(model._device).eval()
        model.sensors, model.controls = list(payload["sensors"]), list(payload["controls"])
        model.time_column = payload.get("time_column")

        def array(value: Optional[torch.Tensor]) -> Optional[np.ndarray]:
            return None if value is None else value.double().numpy()

        sc = payload["scaler"]
        model._mean, model._std, model._vs = array(sc["mean"]), array(sc["std"]), array(sc["vs"])
        model._vmean = array(sc["vmean"])
        model._cmean, model._cstd, model._gain = array(sc["cmean"]), array(sc["cstd"]), array(sc["gain"])
        mask = array(sc.get("level_mask"))
        model._level_mask = np.ones(len(model.sensors)) if mask is None else mask
        model._cal = ScoreCalibration.from_payload(payload["calibration"])
        fc = payload["forecast"]
        model._fc = {"backtested": bool(fc["backtested"]), "weights": array(fc["weights"]), "q80": array(fc["q80"]),
                     "q95": array(fc["q95"]), "skill": array(fc["skill"])}
        if payload.get("fit_summary"):
            model.fit_summary_ = FitSummary.from_dict(payload["fit_summary"])
        tail = payload.get("tail") or {}
        if tail.get("values") is not None:
            model._tail = {
                "values": array(tail["values"]), "controls": array(tail["controls"]),
                "timestamps": tail.get("timestamps"), "time_seconds": array(tail.get("time_seconds")),
                "sample_interval": tail.get("sample_interval"), "rows": int(tail.get("rows", 0)),
            }
        return model


class StreamMonitor:
    """
    Score live telemetry one row at a time.

        monitor = model.stream()
        for row in incoming_rows:                 # dict of {column: value}
            step = monitor.update(row)
            if step.is_anomaly:
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
        self._field: Optional[torch.Tensor] = None
        self._prev_field: Optional[torch.Tensor] = None
        self._prev_q: Optional[np.ndarray] = None
        self._prev_x: Optional[torch.Tensor] = None
        self._prev_a: Optional[torch.Tensor] = None
        self._p_hat: Optional[np.ndarray] = None
        self._score_state: Optional[Dict[str, np.ndarray]] = None
        self._streak = 0

    @torch.no_grad()
    def _advance(self, qn: np.ndarray, p: np.ndarray, un: np.ndarray) -> np.ndarray:
        """
        Consume one normalised row. Returns the velocity that had been predicted for this row
        (zeros for the very first row), then prepares the prediction for the next one.
        """
        model, n = self.model, len(self.model.sensors)
        predicted = np.zeros(n) if self._p_hat is None else self._p_hat
        state = np.concatenate([np.clip(qn, -_CLIP, _CLIP) * model._level_mask, np.clip(p, -_CLIP, _CLIP)])
        x = torch.as_tensor(state, dtype=torch.float32, device=model._device).view(1, 1, -1)
        a = torch.as_tensor(np.clip(un, -_CLIP, _CLIP), dtype=torch.float32, device=model._device).view(1, 1, -1)
        if self.adapt and self._prev_x is not None:
            # The true next state is known now, so let the engine learn from the transition it
            # just mispredicted. This happens *after* its prediction was recorded: no leakage.
            self.engine(self._prev_x, self._prev_a, standing_field=self._prev_field, plastic_runtime=True, y_target=x)
        self._prev_field = self._field
        out = self.engine(x, a, standing_field=self._field, plastic_runtime=False)
        self._field = out["standing_field"]
        self._p_hat = out["output"][0, 0, n:2 * n].double().cpu().numpy()
        self._prev_x, self._prev_a = x, a
        return predicted

    def update(self, row: Any, controls: Any = None, timestamp: Optional[str] = None) -> StepResult:
        """
        Feed the next reading. `row` is a dict {column: value} (extra keys are ignored) or a
        sequence of sensor values in the model's sensor order.
        """
        model = self.model
        n = len(model.sensors)
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
            values = np.asarray(row, dtype=float).reshape(-1)
            if values.shape[0] != n:
                raise DataError(f"Expected {n} sensor values, got {values.shape[0]}.")
            ctrl = np.full(len(model.controls), math.nan) if controls is None else np.asarray(controls, dtype=float).reshape(-1)
        if np.isnan(values).any():
            if self._prev_q is None:
                raise DataError("The first reading of a stream cannot contain missing values.")
            values = np.where(np.isnan(values), self._prev_q * model._std + model._mean, values)

        previous = self._prev_q
        qn, dq, p, un = model._normalise(values[None, :], ctrl[None, :], previous=previous)
        qn, dq, p, un = qn[0], dq[0], p[0], un[0]
        p_hat = self._advance(qn, p, un)

        first = previous is None
        residual = np.clip(p - model._gain * p_hat, -_Z_CAP, _Z_CAP)[None, :]
        scored = score_residuals(residual, level=qn[None, :], delta=dq[None, :], cal=model._cal,
                                 state=self._score_state, valid=np.array([not first]), range_check=not self.adapt)
        self._score_state = scored.state
        sensor_score = np.minimum(scored.sensor_score[0], _Z_CAP)
        score = float(min(scored.score[0], _Z_CAP))
        expected_qn = qn if first else previous + model._vmean + model._vs * model._gain * p_hat
        expected = expected_qn * model._std + model._mean
        self._prev_q = qn

        flagged = []
        for i in np.argsort(-sensor_score):
            if sensor_score[i] <= self.threshold:
                break
            reason = REASONS[int(scored.reason[0, i])]
            signed = scored.z[0, i] if reason == "surprise" else scored.drift_z[0, i]
            direction = "above" if signed >= 0 else "below"
            if reason == "flatline":
                detail = f"reading stuck at {_fmt(values[i])}"
            elif reason == "range":
                direction = "above" if values[i] >= model._mean[i] else "below"
                detail = f"{direction} the range seen in training ({_fmt(values[i])})"
            else:
                detail = f"{REASON_TEXT[reason]} {direction} expected ({_fmt(values[i])} vs {_fmt(expected[i])})"
            flagged.append({"name": model.sensors[i], "score": round(float(sensor_score[i]), 3),
                            "reason": REASON_TEXT[reason], "direction": direction, "detail": detail})
        is_anomaly = score > self.threshold
        self._streak = self._streak + 1 if is_anomaly else 0
        result = StepResult(
            index=self.index, time=timestamp or f"step {self.index}", score=score, is_anomaly=is_anomaly,
            alert=is_anomaly and (self._streak >= 2 or score >= 1.25 * self.threshold),
            sensors=flagged,
            expected={s: float(v) for s, v in zip(model.sensors, expected)},
            values={s: float(v) for s, v in zip(model.sensors, values)},
            sensor_scores={s: float(v) for s, v in zip(model.sensors, sensor_score)},
        )
        self.index += 1
        return result
