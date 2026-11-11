"""
RCWE in five lines: learn normal behaviour, find anomalies, forecast, save.

    python examples/01_quickstart.py

Uses the built-in pump-station demo data, so it runs anywhere. Results are written to
./rcwe_output/example_quickstart/.
"""
from pathlib import Path

from rcwe import TelemetryModel
from rcwe.toolkit.demo import write_demo_files

out = Path("rcwe_output/example_quickstart")
paths = write_demo_files(out)                          # pump_station_normal.csv + pump_station_incident.csv

model = TelemetryModel().fit(paths["normal"])          # 1. learn what "normal" looks like
print(model.summary(), "\n")

result = model.detect(paths["incident"])               # 2. scan a new log
print(result.summary(), "\n")

for event in result.events:                            # every event says when, which sensor, and how it deviated
    lead = event.sensors[0]
    print(f"  event {event.index}: {event.kind:<20} {lead['name']:<16} "
          f"{lead['actual']:.2f} vs {lead['expected']:.2f} expected ({event.severity})")

forecast = model.forecast(steps=48, data=paths["incident"])   # 3. look ahead, with intervals
print("\n" + forecast.summary())

result.forecast = forecast                             # include the forecast in the report
print("\nreport:", result.to_html(out / "report.html", title="Pump station: incident scan"))
print("scores:", result.to_csv(out / "scores.csv"))
print("model: ", model.save(out / "pump_station.rcwe"))     # 4. reuse later with TelemetryModel.load(...)
