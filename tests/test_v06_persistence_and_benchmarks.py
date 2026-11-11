import torch
from pathlib import Path
from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.predict_cli import run_universal_prediction
from rcwe.benchmarks.real_world_suite import RealWorldBenchmarkSuite
from rcwe.server.live_server import LiveRCWEController


def test_checkpoint_save_and_load_pretrained_state_fidelity(tmp_path: Path):
    torch.manual_seed(301)
    engine = ResonantCausalWorldEngine(d_input=32, d_model=64, d_action=8, num_worldlines=3)
    x = torch.randn(2, 6, 32)
    a = torch.randn(2, 6, 8)
    y = x + 0.1

    out1 = engine(x, a, plastic_runtime=True, y_target=y)
    sf1 = out1["standing_field"]

    ckpt_file = tmp_path / "rcwe_test_ckpt.pt"
    engine.save_pretrained(ckpt_file, standing_field=sf1)
    assert ckpt_file.exists()

    loaded_engine, loaded_sf = ResonantCausalWorldEngine.load_pretrained(ckpt_file, device="cpu")
    assert loaded_sf is not None
    assert torch.allclose(sf1.cpu(), loaded_sf.cpu(), atol=1e-6)
    assert torch.allclose(
        engine.metabolic_synapse.fast_weight.cpu(),
        loaded_engine.metabolic_synapse.fast_weight.cpu(),
        atol=1e-6,
    )
    assert torch.allclose(engine.fast_sensory_bias.cpu(), loaded_engine.fast_sensory_bias.cpu(), atol=1e-6)

    with torch.no_grad():
        out_orig = engine(x, a, standing_field=sf1, plastic_runtime=False)["output"]
        out_load = loaded_engine(x.cpu(), a.cpu(), standing_field=loaded_sf, plastic_runtime=False)["output"]
    assert torch.allclose(out_orig.cpu(), out_load.cpu(), atol=1e-5)


def test_universal_cli_auto_schema_and_live_controller(tmp_path: Path):
    # Verify LiveRCWEController step, anomaly injection, and sleep consolidation
    ctrl = LiveRCWEController()
    t1 = ctrl.step_tick()
    assert t1["step"] == 1
    anom = ctrl.inject_anomaly(regime=1)
    assert anom["regime"] == 1
    t2 = ctrl.step_tick()
    assert t2["regime"] == 1
    slp = ctrl.trigger_sleep_consolidation()
    assert slp["after_fast_norm"] == 0.0


def test_real_world_multi_domain_benchmark_suite(tmp_path: Path):
    rep = RealWorldBenchmarkSuite.run_all_benchmarks(out_dir=str(tmp_path / "rw_data"))
    assert len(rep["benchmarks"]) == 3
    # The sensors each generated log adds its fault to (see generate_benchmark_csvs).
    injected = {
        "ETT_Transformer_Thermal": {"q_2", "q_3"},
        "SWaT_CyberPhysical_Water": {"q_4", "q_5"},
        "NASA_Turbofan_Bearing": {"q_7", "q_8"},
    }
    for b in rep["benchmarks"]:
        assert b["rcwe_post_shift_mse"] < b["tf_post_shift_mse"]
        assert b["improvement_ratio"] > 1.5
        assert b["flagged_sensors"] and b["flagged_sensors"][0] in injected[b["domain"]]
