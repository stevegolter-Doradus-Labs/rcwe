"""Mission Control: the page, and the live engine that feeds it real numbers."""
from pathlib import Path

import pytest

import rcwe
from rcwe.ui.mission_control import HTML_TEMPLATE, MissionControlEngine


@pytest.fixture(scope="module")
def engine():
    return MissionControlEngine()


def test_snapshot_reports_measured_state_sizes_and_version(engine):
    snap = engine.get_snapshot()
    assert snap["live"] is True and snap["version"] == rcwe.__version__
    assert snap["psi_kb"] == 8.0            # measured from the standing-wavefield tensor: 16 bands x 64 complex64
    assert snap["swarm_kb"] == 32.0         # four agents x 8 KB
    assert snap["regime"] == 0 and snap["regime_label"] == "W0 NOMINAL"
    assert len(snap["psi"]["mag"]) == 16 and len(snap["thermal"]["grid"]) == 16 and len(snap["core"]["posteriors"]) == 3
    assert any("calibrated" in entry["msg"] for entry in snap["log"])


def test_ticks_advance_real_models_and_beat_the_frozen_baseline(engine):
    first = engine.tick()
    rows = [engine.tick() for _ in range(40)]
    assert rows[-1]["step"] == first["step"] + 40
    rcwe_err = sum(r["core"]["rcwe_mse"] for r in rows) / len(rows)
    tf_err = sum(r["core"]["tf_mse"] for r in rows) / len(rows)
    assert 0.0 < rcwe_err < tf_err                                   # scored before the engine sees the answer
    assert abs(sum(rows[-1]["core"]["posteriors"]) - 1.0) < 1e-4
    assert rows[-1]["kpi"]["rcwe_mse"] is not None and rows[-1]["scan_ms"] > 0


def test_operator_actions_change_the_simulation(engine):
    fault = engine.trigger_action("scada_fault")
    assert fault["status"] == "ok" and fault["action"] == "scada_fault"
    assert engine.get_snapshot()["regime"] == 1
    assert float(engine.sandbox.sink_mask[5]) == pytest.approx(0.45)   # the sandbox topology really changed
    for _ in range(50):
        snap = engine.tick()
    assert snap["core"]["posteriors"][1] > 0.5                       # ...and the engine identifies the sinkhole regime

    engine.trigger_action("thermal_hotspot")
    assert engine.tick()["thermal"]["peak"] > 2.0
    painted = engine.trigger_action("paint", row=99, col=-4)
    assert painted["row"] == 15 and painted["col"] == 0             # clamped to the 16x16 grid

    engine.trigger_action("swarm_cascade")
    for _ in range(12):
        snap = engine.tick()
    assert snap["swarm"]["fault"] == [True, True, True, True]
    assert snap["kpi"]["cascade_coupled"] < snap["kpi"]["cascade_isolated"]   # gossip helps at first impact...
    assert snap["kpi"]["precost_coupled"] > 0.1                              # ...at a measurable cost before it

    engine.trigger_action("rl_shock")
    assert engine.tick()["rl"]["shock"] is True

    slept = engine.trigger_action("sleep")
    assert slept["status"] == "ok" and slept["consolidated_norm"] > 0
    assert float(engine.rcwe.metabolic_synapse.fast_weight.abs().sum()) == 0.0

    assert engine.trigger_action("restore")["status"] == "ok" and engine.get_snapshot()["regime"] == 0
    bogus = engine.trigger_action("self_destruct")
    assert bogus["status"] == "error" and "unknown action" in bogus["error"]


def test_standalone_page_is_an_honestly_labelled_simulation(tmp_path: Path):
    out_file = tmp_path / "mission_control.html"
    resolved = MissionControlEngine.export_standalone_html(str(out_file))
    html = Path(resolved).read_text(encoding="utf-8")
    assert html == HTML_TEMPLATE
    assert "DORADUS LABS — RCWE MISSION CONTROL" in html
    assert "Track 1: 2D Spectral Thermal Array" in html
    assert "Track 2: Symplectic RL Phase-Space" in html
    assert "Track 3: 4-Substation Wavefield Swarm" in html
    assert 'id="simBanner"' in html and "nothing shown is a measurement" in html
    for stale in ("293.11", "160.1x", "14.03x", "33,554,432", "MPS Metal MSL Ready"):
        assert stale not in html                                     # no benchmark figure is baked into the page
