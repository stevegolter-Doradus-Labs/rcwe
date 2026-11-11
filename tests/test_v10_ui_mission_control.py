from pathlib import Path
from rcwe.ui.mission_control import MissionControlEngine, HTML_TEMPLATE


def test_v10_mission_control_engine_snapshot_and_actions():
    mc = MissionControlEngine()
    snap = mc.get_snapshot()
    assert snap["version"] == "1.0.0"
    assert snap["psi_kb"] == 8.0
    assert snap["swarm_kb"] == 32.0

    r1 = mc.trigger_action("scada_fault")
    assert r1["status"] == "ok" and r1["action"] == "scada_fault"

    r2 = mc.trigger_action("sleep")
    assert r2["status"] == "ok" and "consolidated_norm" in r2


def test_v10_mission_control_standalone_html_export(tmp_path: Path):
    out_file = tmp_path / "rcwe_mission_control_v10.html"
    resolved = MissionControlEngine.export_standalone_html(str(out_file))
    assert Path(resolved).exists()
    html = Path(resolved).read_text(encoding="utf-8")
    assert "DORADUS LABS — RCWE v1.0.0 MISSION CONTROL" in html
    assert "Track 1: 2D Spectral Thermal Array" in html
    assert "Track 2: Symplectic RL Phase-Space" in html
    assert "Track 3: 4-Substation Wavefield Swarm" in html
