"""
RCWE Toolkit -- the end-user layer on top of the Resonant Causal World Engine.

    from rcwe import TelemetryModel
    model = TelemetryModel().fit("normal.csv")
    print(model.detect("today.csv").summary())
"""
from rcwe.toolkit.table import TelemetryTable, load_table, DataError
from rcwe.toolkit.results import AnomalyEvent, DetectionResult, Forecast, FitSummary
from rcwe.toolkit.model import TelemetryModel, StreamMonitor, StepResult, NotFittedError
from rcwe.toolkit.demo import make_pump_station, write_demo_files

__all__ = [
    "TelemetryModel", "StreamMonitor", "StepResult", "NotFittedError",
    "TelemetryTable", "load_table", "DataError",
    "AnomalyEvent", "DetectionResult", "Forecast", "FitSummary",
    "make_pump_station", "write_demo_files",
]
