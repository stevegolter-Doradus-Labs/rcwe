"""
Point RCWE at your own file.

    python examples/02_your_own_csv.py path/to/log.csv
    python examples/02_your_own_csv.py path/to/log.csv --train path/to/known_good.csv

Without --train, normal behaviour is learned from the first 60% of the log. The script
first shows how the file was understood (which columns are sensors, controls, time), so
you can correct that before trusting the result.
"""
import argparse
from pathlib import Path

from rcwe import TelemetryModel, load_table

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("log", help="CSV / TSV / JSON / JSON-lines telemetry file")
parser.add_argument("--train", help="separate log of known-good operation")
parser.add_argument("--sensitivity", default="medium", help="low | medium | high")
args = parser.parse_args()

# ---- 1. see how the file is read ------------------------------------------------------
table = load_table(args.log)        # pass sensors=[...], controls=[...], time_column="..." to override
info = table.describe()
print(f"{info['rows']:,} rows from {info['start']} to {info['end']}")
print("sensors :", ", ".join(table.sensors))
print("controls:", ", ".join(table.controls) or "(none)")
for note in table.notes:
    print("note    :", note)

# ---- 2. learn normal behaviour --------------------------------------------------------
if args.train:
    model = TelemetryModel().fit(args.train)
else:
    head, _ = table.split(0.6)
    model = TelemetryModel().fit(head)
print("\n" + model.summary())

# ---- 3. scan, explain, export ---------------------------------------------------------
result = model.detect(table, sensitivity=args.sensitivity)
print("\n" + result.summary())

out = Path("rcwe_output") / Path(args.log).stem
print("\nreport:", result.to_html(out / "report.html", title=f"RCWE analysis: {Path(args.log).name}"))
print("events:", result.to_json(out / "result.json", include_series=False))
print("model: ", model.save(out / "model.rcwe"))
