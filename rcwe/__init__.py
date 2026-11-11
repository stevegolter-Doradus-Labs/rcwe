"""
Resonant Causal World Engine (RCWE).

For most users the entry point is the toolkit layer:

    from rcwe import TelemetryModel

    model = TelemetryModel().fit("normal_operation.csv")
    print(model.detect("today.csv").summary())

The research engine itself (`ResonantCausalWorldEngine`) and the benchmark harnesses remain
available for anyone who wants to work at that level.
"""
from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.sandbox.cpfn_env import CyberPhysicalFlowSandbox
from rcwe.sandbox.baseline_tf import BaselineCausalTransformer
from rcwe.diagnostics.nl_codec import CausalGraphNLCodec, CausalDiagnosticPacket
from rcwe.diagnostics.profiler import HardwareProfilingHarness
from rcwe.diagnostics.validation import ScientificValidationBenchmark
from rcwe.kernels.accelerator import select_best_device, sync_device, get_device_memory_kb

__version__ = "1.1.0"

from rcwe.toolkit import (  # noqa: E402  (needs __version__ to be defined first)
    AnomalyEvent,
    DataError,
    DetectionResult,
    FitSummary,
    Forecast,
    NotFittedError,
    StepResult,
    StreamMonitor,
    TelemetryModel,
    TelemetryTable,
    load_table,
    make_pump_station,
    write_demo_files,
)

__all__ = [
    # end-user toolkit
    "TelemetryModel",
    "StreamMonitor",
    "StepResult",
    "TelemetryTable",
    "load_table",
    "DetectionResult",
    "AnomalyEvent",
    "Forecast",
    "FitSummary",
    "DataError",
    "NotFittedError",
    "make_pump_station",
    "write_demo_files",
    # research engine & harnesses
    "ResonantCausalWorldEngine",
    "CyberPhysicalFlowSandbox",
    "BaselineCausalTransformer",
    "CausalGraphNLCodec",
    "CausalDiagnosticPacket",
    "HardwareProfilingHarness",
    "ScientificValidationBenchmark",
    "select_best_device",
    "sync_device",
    "get_device_memory_kb",
]
