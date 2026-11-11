"""
Score readings one at a time, the way a live feed delivers them.

    python examples/03_live_stream.py
    python examples/03_live_stream.py --adapt      # keep learning: flag when a change starts, then accept it

`model.stream()` returns a monitor; feed it each new row (a dict) and it answers
immediately. Here the "live feed" is the demo incident log replayed row by row.
For a real feed that is appended to a file, `rcwe watch file.csv -m model.rcwe` does this for you.
"""
import argparse

from rcwe import TelemetryModel
from rcwe.toolkit.demo import split_demo

parser = argparse.ArgumentParser()
parser.add_argument("--adapt", action="store_true", help="online adaptation (the engine's plasticity)")
parser.add_argument("--rows", type=int, default=1200, help="how many rows of the feed to replay")
args = parser.parse_args()

normal, incident, _ = split_demo()
model = TelemetryModel().fit(normal)
monitor = model.stream(sensitivity="medium", adapt=args.adapt)

in_alarm = False
for row in incident.to_rows()[: args.rows]:
    step = monitor.update(row)                      # <- one reading in, one verdict out
    if step.alert and not in_alarm:
        worst = step.sensors[0]
        print(f"{step.time}  ALERT   score {step.score:5.1f}  {worst['name']}: {worst['detail']}")
        in_alarm = True
    elif in_alarm and not step.is_anomaly:
        print(f"{step.time}  cleared")
        in_alarm = False

print(f"\nprocessed {monitor.index:,} readings in {'adaptive' if args.adapt else 'reference'} mode")
