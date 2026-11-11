"""
Turning prediction residuals into anomaly scores, events and explanations.

The world model predicts the next reading of every sensor. Scoring compares what actually
arrived with what was expected and expresses the difference in calibrated "sigma" units,
using four complementary signals per sensor:

- surprise   instantaneous one-step residual                   -> spikes and jumps
- drift      EWMA of the residuals (a control chart)           -> small but persistent bias
- range      distance outside the envelope seen in training    -> the model is extrapolating
- flatline   run length of an unchanging reading               -> stuck / frozen sensor

Every signal is scaled so that `NOMINAL_LEVEL` means "alarm at default sensitivity", which
keeps a single threshold meaningful across signals and sensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "NOMINAL_LEVEL", "SENSITIVITY_LEVELS", "REASONS", "ScoreCalibration", "SensorScores",
    "resolve_threshold", "robust_sigma", "ewma", "score_residuals", "find_events",
    "longest_run", "detection_metrics",
]

NOMINAL_LEVEL = 4.5
SENSITIVITY_LEVELS = {"low": 6.0, "medium": NOMINAL_LEVEL, "high": 3.5}
REASONS = ("surprise", "drift", "range", "flatline")
REASON_TEXT = {
    "surprise": "sudden jump",
    "drift": "sustained deviation",
    "range": "outside the range seen in training",
    "flatline": "reading stuck at a constant value",
}
_GAUSS_Q99 = 2.5758
_EWMA_INPUT_LIMIT = 25.0


def resolve_threshold(sensitivity: Any) -> float:
    """Map 'low' | 'medium' | 'high' (or a number of sigmas) to a score threshold."""
    if isinstance(sensitivity, str):
        key = sensitivity.strip().lower()
        if key not in SENSITIVITY_LEVELS:
            raise ValueError(f"sensitivity must be one of {sorted(SENSITIVITY_LEVELS)} or a number, got '{sensitivity}'.")
        return SENSITIVITY_LEVELS[key]
    value = float(sensitivity)
    if not value > 0:
        raise ValueError("sensitivity must be a positive number of sigmas.")
    return value


def robust_sigma(residuals: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """
    Per-column scale that is robust to outliers yet honest about heavy tails: the larger of
    the MAD-based sigma and the sigma implied by the empirical 99th percentile.
    """
    res = np.asarray(residuals, dtype=float)
    centered = res - np.median(res, axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(centered), axis=0)
    sigma = mad
    if res.shape[0] >= 200:
        q99 = np.quantile(np.abs(centered), 0.99, axis=0)
        sigma = np.maximum(mad, q99 / _GAUSS_Q99)
    elif res.shape[0] >= 8:
        sigma = np.maximum(mad, np.std(res, axis=0))
    return np.maximum(sigma, floor)


def ewma(values: np.ndarray, lam: float, initial: Optional[np.ndarray] = None) -> np.ndarray:
    """Exponentially weighted moving average along axis 0."""
    out = np.empty_like(values, dtype=float)
    state = np.zeros(values.shape[1:], dtype=float) if initial is None else np.asarray(initial, dtype=float).copy()
    for t in range(values.shape[0]):
        state = lam * values[t] + (1.0 - lam) * state
        out[t] = state
    return out


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values in a 1-D boolean array."""
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        if run > best:
            best = run
    return best


@dataclass
class ScoreCalibration:
    """Everything needed to convert raw residuals into calibrated scores (per sensor)."""

    sigma: np.ndarray                 # residual scale, in velocity-normalised units
    ewma_sigma: np.ndarray            # scale of the EWMA statistic
    ewma_lambda: float = 0.25
    range_lo: Optional[np.ndarray] = None      # training envelope, level-normalised units
    range_hi: Optional[np.ndarray] = None
    range_margin: Optional[np.ndarray] = None  # distance outside the envelope that maps to NOMINAL_LEVEL
    range_enabled: Optional[np.ndarray] = None
    flat_eps: Optional[np.ndarray] = None      # |change| at or below this counts as "no change"
    flat_run: Optional[np.ndarray] = None      # run length that maps to NOMINAL_LEVEL (0 = disabled)

    def to_payload(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ewma_lambda": float(self.ewma_lambda)}
        for name in ("sigma", "ewma_sigma", "range_lo", "range_hi", "range_margin", "range_enabled", "flat_eps", "flat_run"):
            value = getattr(self, name)
            out[name] = None if value is None else [float(v) for v in np.asarray(value, dtype=float)]
        return out

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ScoreCalibration":
        def arr(key: str) -> Optional[np.ndarray]:
            value = payload.get(key)
            return None if value is None else np.asarray(value, dtype=float)
        return cls(
            sigma=arr("sigma"), ewma_sigma=arr("ewma_sigma"), ewma_lambda=float(payload.get("ewma_lambda", 0.25)),
            range_lo=arr("range_lo"), range_hi=arr("range_hi"), range_margin=arr("range_margin"),
            range_enabled=arr("range_enabled"), flat_eps=arr("flat_eps"), flat_run=arr("flat_run"),
        )


@dataclass
class SensorScores:
    """Per-step, per-sensor scoring output."""

    z: np.ndarray            # signed instantaneous residual, sigma units            [T, n]
    drift_z: np.ndarray      # signed EWMA statistic, sigma units                    [T, n]
    range_z: np.ndarray      # range-guard signal                                    [T, n]
    flat_z: np.ndarray       # flatline signal                                       [T, n]
    sensor_score: np.ndarray # max of the four signals                               [T, n]
    reason: np.ndarray       # index into REASONS of the dominant signal             [T, n]
    score: np.ndarray        # worst sensor at each step                             [T]
    state: Dict[str, np.ndarray] = field(default_factory=dict)  # carry-over for streaming


def score_residuals(
    residual: np.ndarray,
    level: np.ndarray,
    delta: np.ndarray,
    cal: ScoreCalibration,
    state: Optional[Dict[str, np.ndarray]] = None,
    valid: Optional[np.ndarray] = None,
    range_check: bool = True,
) -> SensorScores:
    """
    Score a block of consecutive steps.

    residual : actual - expected next-step change, velocity-normalised      [T, n]
    level    : actual sensor level, level-normalised                         [T, n]
    delta    : actual change since the previous step, level-normalised       [T, n]
    state    : carry-over from the previous block (None for the start of a sequence)
    valid    : optional [T] mask; rows marked False (e.g. the very first row, which has no
               prediction) contribute nothing to the residual-based signals.
    """
    residual = np.asarray(residual, dtype=float)
    T, n = residual.shape
    if valid is not None:
        residual = np.where(valid[:, None], residual, 0.0)

    z = residual / cal.sigma
    prev = None if state is None else state.get("ewma")
    # Winsorise what feeds the drift chart: one wild reading (a sensor glitch) should register as
    # a spike, not keep the drift alarm ringing for dozens of steps afterwards.
    limit = _EWMA_INPUT_LIMIT * cal.sigma
    smooth = ewma(np.clip(residual, -limit, limit), cal.ewma_lambda, initial=prev)
    drift_z = smooth / cal.ewma_sigma

    range_z = np.zeros((T, n))
    if range_check and cal.range_lo is not None and cal.range_enabled is not None:
        outside = np.maximum(level - cal.range_hi, 0.0) + np.maximum(cal.range_lo - level, 0.0)
        # Capped: "how far outside" is not on the same footing as a residual in sigmas, and should
        # not outrank sharp model-based evidence when sensors are ordered by blame.
        range_z = np.minimum(NOMINAL_LEVEL * outside / cal.range_margin, 3.0 * NOMINAL_LEVEL) * (cal.range_enabled > 0)

    flat_z = np.zeros((T, n))
    runs = np.zeros(n) if state is None or "flat" not in state else state["flat"].astype(float).copy()
    if cal.flat_run is not None and cal.flat_eps is not None:
        enabled = cal.flat_run > 0
        still = np.abs(delta) <= cal.flat_eps
        if valid is not None:
            still = still & valid[:, None]
        safe_run = np.where(enabled, cal.flat_run, 1.0)
        for t in range(T):
            runs = np.where(still[t], runs + 1.0, 0.0)
            flat_z[t] = np.where(enabled, np.minimum(NOMINAL_LEVEL * runs / safe_run, 4.0 * NOMINAL_LEVEL), 0.0)

    stacked = np.stack([np.abs(z), np.abs(drift_z), range_z, flat_z], axis=0)
    sensor_score = stacked.max(axis=0)
    reason = stacked.argmax(axis=0)
    score = sensor_score.max(axis=1) if n else np.zeros(T)
    new_state = {"ewma": smooth[-1].copy() if T else (np.zeros(n) if prev is None else prev), "flat": runs}
    return SensorScores(z=z, drift_z=drift_z, range_z=range_z, flat_z=flat_z,
                        sensor_score=sensor_score, reason=reason, score=score, state=new_state)


def find_events(
    score: np.ndarray,
    threshold: float,
    merge_gap: int = 3,
    min_duration: int = 1,
    lone_point_factor: float = 1.25,
    exit_factor: float = 0.6,
) -> List[Tuple[int, int]]:
    """
    Group threshold exceedances into events, returned as inclusive (start, end) index pairs.

    Two exceedances belong to the same event when at most `merge_gap` quiet steps separate
    them, or when the score never falls below `exit_factor * threshold` in between
    (hysteresis: an incident whose score dips but stays elevated is still one incident).
    A one-off single-step blip must clear `lone_point_factor * threshold` to count, which
    suppresses most chance exceedances without hiding genuine spikes.
    """
    score = np.asarray(score, dtype=float)
    flagged = np.flatnonzero(score > threshold)
    if flagged.size == 0:
        return []
    exit_level = exit_factor * threshold
    events: List[Tuple[int, int]] = []
    start = prev = int(flagged[0])
    for idx in flagged[1:]:
        idx = int(idx)
        gap = idx - prev - 1
        if gap > merge_gap and not np.all(score[prev + 1:idx] > exit_level):
            events.append((start, prev))
            start = idx
        prev = idx
    events.append((start, prev))

    kept = []
    for a, b in events:
        hits = int(np.sum(score[a:b + 1] > threshold))
        if hits < max(min_duration, 1):
            continue
        if hits == 1 and float(np.max(score[a:b + 1])) < lone_point_factor * threshold:
            continue
        kept.append((a, b))
    return kept


def _label_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(labels > 0)
    if idx.size == 0:
        return []
    segments = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i != prev + 1:
            segments.append((start, prev))
            start = i
        prev = i
    segments.append((start, prev))
    return segments


def detection_metrics(
    labels: np.ndarray,
    score: np.ndarray,
    events: Sequence[Tuple[int, int]],
    tolerance: int = 3,
) -> Dict[str, Any]:
    """
    Compare detections with ground-truth labels.

    Event-level numbers answer the operational questions ("was each real incident caught,
    how quickly, and how many false alarms were raised"); point-wise precision/recall/F1 and
    AUROC are included for comparison with published benchmarks.
    """
    labels = (np.asarray(labels) > 0).astype(int)
    T = labels.shape[0]
    flagged = np.zeros(T, dtype=bool)
    for a, b in events:
        flagged[a:b + 1] = True
    tp = int(np.sum(flagged & (labels == 1)))
    fp = int(np.sum(flagged & (labels == 0)))
    fn = int(np.sum(~flagged & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    segments = _label_segments(labels)
    delays: List[int] = []
    caught = 0
    for a, b in segments:
        window = flagged[a:min(b + tolerance, T - 1) + 1]
        if window.any():
            caught += 1
            delays.append(int(np.argmax(window)))
    false_alarms = 0
    for a, b in events:
        lo, hi = max(a - tolerance, 0), min(b + tolerance, T - 1)
        if not labels[lo:hi + 1].any():
            false_alarms += 1

    auroc: Optional[float] = None
    positives, negatives = int(labels.sum()), int(T - labels.sum())
    if positives and negatives:
        order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
        ranked = labels[order]
        tpr = np.concatenate([[0.0], np.cumsum(ranked) / positives])
        fpr = np.concatenate([[0.0], np.cumsum(1 - ranked) / negatives])
        auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))

    return {
        "labeled_incidents": len(segments),
        "incidents_detected": caught,
        "incident_recall": caught / len(segments) if segments else None,
        "false_alarm_events": false_alarms,
        "median_detection_delay_steps": float(np.median(delays)) if delays else None,
        "point_precision": precision,
        "point_recall": recall,
        "point_f1": f1,
        "auroc": auroc,
        "labeled_anomalous_points": positives,
    }
