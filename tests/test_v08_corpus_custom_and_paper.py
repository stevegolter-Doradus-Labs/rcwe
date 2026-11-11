from pathlib import Path
from rcwe.lm.corpus_trainer import LargeCorpusRCWETrainer
from rcwe.ingestion.custom_connector import CustomDatasetConnector


def test_large_corpus_rcwe_deep_trainer_and_generation(tmp_path: Path):
    trainer = LargeCorpusRCWETrainer(d_model=64, num_layers=2, seq_len=128)
    rep = trainer.train_and_evaluate(
        epochs=12,
        batch_size=4,
        save_ckpt=str(tmp_path / "deep_lm.pt"),
    )
    assert rep["corpus_total_chars"] > 1000
    assert rep["rcwe_deep_bpb"] > 0.0
    assert len(rep["sample_completion"]) > 20
    assert (tmp_path / "deep_lm.pt").exists()


def test_custom_dataset_connector_scada_substation(tmp_path: Path):
    rep = CustomDatasetConnector.run_custom_directory_scan(custom_dir=str(tmp_path / "custom_ds"))
    assert rep["inferred_sensors"] == 6
    assert rep["rcwe_tail_mse"] < rep["tf_tail_mse"]
    # The template faults five of its six sensors at once; reactive_MVAR stays healthy and must not be blamed.
    faulted = {"voltage_bus_A_kV", "voltage_bus_B_kV", "current_line_1_A", "freq_Hz", "transformer_oil_C"}
    assert rep["flagged_anomalous_sensors"] and set(rep["flagged_anomalous_sensors"]) <= faulted
    assert "also involved" in rep["anomaly_events"][0]          # the other faulted sensors are named, not just the sharpest


def test_arxiv_latex_preprint_exists_and_well_formed():
    tex_path = Path("paper/Doradus_RCWE_arXiv_Preprint_v08.tex")
    assert tex_path.exists()
    content = tex_path.read_text(encoding="utf-8")
    assert "\\documentclass" in content
    assert "\\begin{document}" in content
    assert "\\end{document}" in content
