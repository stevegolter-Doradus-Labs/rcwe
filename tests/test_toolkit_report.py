"""HTML report: self-contained, driven by real result data, and safe with hostile column names."""
import json
import re

import numpy as np

from rcwe.toolkit.report import build_report_data, render_report
from rcwe.toolkit.results import AnomalyEvent, DetectionResult, Forecast
from rcwe.toolkit.scoring import find_events


def _result(rows=600, sensors=("flow", "pressure"), labels=True):
    rng = np.random.default_rng(0)
    n = len(sensors)
    values = rng.normal(size=(rows, n)).cumsum(axis=0) * 0.05 + 10.0
    expected = values + rng.normal(scale=0.02, size=(rows, n))
    sensor_scores = np.abs(rng.normal(size=(rows, n)))
    sensor_scores[200:230, 1] = 9.0
    scores = sensor_scores.max(axis=1)
    spans = find_events(scores, 4.5)
    stamps = [f"2026-01-01T00:{t // 60:02d}:{t % 60:02d}Z" for t in range(rows)]
    events = [
        AnomalyEvent(index=k, start=a, end=b, start_time=stamps[a], end_time=stamps[b], peak=a, peak_time=stamps[a],
                     peak_score=float(scores[a:b + 1].max()), severity="high", kind="sustained deviation",
                     sensors=[{"name": sensors[1], "peak_score": 9.0, "share": 1.0, "reason": "drift", "direction": "above",
                               "peak_time": stamps[a], "actual": 11.0, "expected": 10.0}],
                     description=f"{stamps[a]} to {stamps[b]}: {sensors[1]} ran above expectations.",
                     what=f"{sensors[1]} ran above expectations.")
        for k, (a, b) in enumerate(spans, 1)
    ]
    truth = np.zeros(rows, dtype=int)
    truth[200:230] = 1
    return DetectionResult(
        sensors=list(sensors), timestamps=stamps, values=values, expected=expected, band=np.full(n, 0.1),
        residual_z=rng.normal(size=(rows, n)), sensor_scores=sensor_scores, scores=scores, threshold=4.5, events=events,
        labels=truth if labels else None, source="unit-test.csv",
        model_info={"skill_vs_persistence": 0.25, "sensor_skill": {s: 0.25 for s in sensors}, "train_rows": 400,
                    "validation_rows": 100, "device": "CPU", "warnings": [], "notes": ["a note"]},
    )


def _payload(html):
    block = re.search(r'<script type="application/json" id="rcwe-data">(.*?)</script>', html, re.S).group(1)
    return json.loads(block)


def test_report_is_self_contained_and_carries_the_result(tmp_path):
    result = _result()
    path = result.to_html(tmp_path / "out" / "report.html", title="Line 4 <night shift>")
    html = open(path, encoding="utf-8").read()

    assert html.startswith("<!DOCTYPE html>") and "</html>" in html
    assert "<title>Line 4 &lt;night shift&gt;</title>" in html
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', html)        # nothing fetched from the network
    data = _payload(html)
    assert data["title"] == "Line 4 <night shift>" and data["rows"] == 600
    assert data["sensors"] == ["flow", "pressure"] and data["threshold"] == 4.5
    assert len(data["events"]) == 1 and data["events"][0]["start"] == 200 and data["events"][0]["what"]
    assert data["index"] == list(range(600)) and len(data["score"]) == 600
    assert data["panels"][0]["name"] == "pressure"                         # most involved sensor first
    assert data["panels"][0]["flagged"] == [[200, 229]]
    assert data["ranking"][0] == {"name": "pressure", "steps": 30, "peak": 9.0}
    assert sum(data["labels"]) == 30 and data["forecast"] is None


def test_report_escapes_hostile_column_names():
    evil = '</script><script>alert("x")</script>'
    html = render_report(_result(sensors=("ok", evil), labels=False))
    assert evil not in html and "<script>alert" not in html
    assert html.count("</script>") == 2                                    # only the report's own two script blocks
    assert _payload(html)["sensors"][1] == evil                            # ...yet the name survives as data


def test_long_logs_are_thinned_but_keep_every_event_in_full():
    result = _result(rows=30000)
    data = build_report_data(result)
    kept = data["index"]
    assert len(kept) < 6000 and kept[0] == 0 and kept[-1] == 29999
    assert set(range(200, 230)) <= set(kept)                               # the event itself is embedded at full detail
    assert max(data["score"]) == round(float(result.scores.max()), 4)      # the peak is never thinned away
    assert len(render_report(result)) < 1_500_000


def test_report_includes_forecast_when_attached():
    result = _result(labels=False)
    steps, n = 12, 2
    centre = np.linspace(10, 11, steps)[:, None].repeat(n, axis=1)
    result.forecast = Forecast(
        sensors=list(result.sensors), values=centre, lower=centre - 0.2, upper=centre + 0.2, lower95=centre - 0.4,
        upper95=centre + 0.4, timestamps=[f"2026-01-01T01:{m:02d}:00Z" for m in range(steps)], origin="2026-01-01T00:09:59Z",
        history=result.values[-50:], history_timestamps=result.timestamps[-50:],
        sensor_skill={"flow": 0.5, "pressure": float("nan")}, model_weight={"flow": 0.9, "pressure": 0.0}, notes=["held controls"],
    )
    data = _payload(render_report(result))
    assert data["forecast"]["steps"] == 12 and len(data["forecast"]["panels"]) == 2
    by_name = {p["name"]: p for p in data["forecast"]["panels"]}
    assert by_name["pressure"]["skill"] is None and by_name["pressure"]["weight"] == 0.0   # NaN never reaches the page
    assert len(by_name["flow"]["history"]) == 50 and by_name["flow"]["upper"][0] > by_name["flow"]["lower"][0]


def test_series_keep_enough_digits_to_show_their_own_variation():
    # A latitude that moves in the fifth decimal and a counter that climbs by 0.5 near 1.2 million:
    # at a fixed five significant digits both were drawn as flat lines under a "critical" event.
    result = _result(rows=400, sensors=("energy_kwh_total", "gps_lat_deg"), labels=False)
    steps = np.arange(400)
    result.values[:, 0] = 1234567.8 + 0.5 * steps
    result.values[:, 1] = 52.520008 + 1e-5 * np.sin(steps / 9.0)
    result.expected = result.values.copy()
    data = build_report_data(result)
    panels = {panel["name"]: panel for panel in data["panels"]}
    assert len(set(panels["gps_lat_deg"]["actual"])) > 50 and panels["gps_lat_deg"]["decimals"] >= 6
    assert len(set(panels["energy_kwh_total"]["actual"])) == 400


def test_every_event_and_every_labelled_stretch_survives_thinning():
    rows = 120000
    result = _result(rows=rows, labels=True)
    scores = np.abs(np.random.default_rng(1).normal(size=rows))
    result.events = []
    for k in range(160):                                        # far more events than the detail budget covers
        t = 500 + 740 * k
        scores[t] = 6.0 + 0.05 * k
        result.events.append(AnomalyEvent(index=k + 1, start=t, end=t, start_time=str(t), end_time=str(t), peak=t,
                                          peak_time=str(t), peak_score=float(scores[t]), severity="low", kind="spike"))
    result.scores = scores
    result.labels = np.zeros(rows, dtype=int)
    for k in range(12):                                         # twelve labelled incidents, most of them undetected
        result.labels[10000 * k + 3000:10000 * k + 3020] = 1
    data = build_report_data(result)
    kept = set(data["index"])
    for event in result.events:
        assert {event.start - 1, event.start, event.start + 1} <= kept, event.index     # something to draw when zoomed in
    assert data["label_spans"] == [[10000 * k + 3000, 10000 * k + 3019] for k in range(12)]
    assert len(data["index"]) < 12000


def test_awkward_titles_and_constant_columns_do_not_break_the_page():
    page = render_report(_result(labels=False), title="RCWE analysis: __DATA__.csv")
    assert re.search(r"<title>RCWE analysis: __DATA__\.csv</title>", page)           # the title is text, not the payload
    assert page.count('id="rcwe-data"') == 1
    # The tick loop counts steps instead of adding them (adding 0.1 to 1.7e15 never arrives).
    assert "for (var k = 0; k < 40; k++)" in page and "v += step" not in page
