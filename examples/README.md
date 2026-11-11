# Examples

Three short scripts, each runnable as is. Install RCWE first (`python -m pip install -e .` from the
repository root). Everything they write goes to `./rcwe_output/`.

| Script | Shows | Run time |
|---|---|---|
| [`01_quickstart.py`](01_quickstart.py) | The whole loop on built-in data: train, detect, forecast, report, save. | about 15 s |
| [`02_your_own_csv.py`](02_your_own_csv.py) | Pointing RCWE at your own file, checking how its columns were read, and exporting the results. | depends on the file |
| [`03_live_stream.py`](03_live_stream.py) | Scoring readings one at a time, the way a live feed delivers them; `--adapt` turns on online learning. | about 20 s |

```bash
python examples/01_quickstart.py
```

```bash
python examples/02_your_own_csv.py examples/data/sample_grid_telemetry.csv
```

```bash
python examples/03_live_stream.py --adapt
```

## Sample data

`data/sample_grid_telemetry.csv` and `.json` are the small grid-telemetry logs that shipped with
earlier versions: 40 rows of a 16-node ring, with a fault injected on nodes 5 and 6 halfway
through. They are deliberately tiny, so RCWE reports that the result is indicative only. For a realistic dataset, run:

```bash
rcwe demo-data
```

That writes `rcwe_demo/pump_station_normal.csv` (ten days of healthy operation) and
`rcwe_demo/pump_station_incident.csv` (the next four days, with five labelled faults).

The same steps are available without writing any Python: see the command-line section of the
[User Guide](../docs/USER_GUIDE.md#4-command-line).
