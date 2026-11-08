import torch
from rcwe.core.metabolic import MetabolicPlasticSynapse
from rcwe.core.engine import ResonantCausalWorldEngine
from rcwe.diagnostics.nl_codec import CausalGraphNLCodec


def test_backprop_free_metabolic_plasticity_and_sleep():
    torch.manual_seed(201)
    synapse = MetabolicPlasticSynapse(d_model=32, lr_local=0.05)

    initial_slow = synapse.slow_weight.weight.clone()
    assert torch.all(synapse.fast_weight == 0.0)

    with torch.no_grad():
        for _ in range(5):
            z_in = torch.complex(torch.randn(2, 8, 32) * 1.5, torch.randn(2, 8, 32))
            _ = synapse(z_in, plastic_runtime=True)

    assert float(synapse.fast_weight.abs().sum().item()) > 0.0, "Fast weights failed to adapt under no_grad!"

    synapse.consolidate_sleep_phase(consolidation_rate=0.1)
    assert float(synapse.fast_weight.abs().sum().item()) == 0.0, "Fast weights not cleared after sleep phase!"
    assert not torch.allclose(initial_slow, synapse.slow_weight.weight), "Slow weights failed to consolidate!"


def test_mdo_epistemic_foraging_and_nl_codec():
    torch.manual_seed(202)
    engine = ResonantCausalWorldEngine(d_input=32, d_model=64, d_action=8, num_worldlines=3)
    codec = CausalGraphNLCodec(d_model=64, num_nodes=16)

    x_seq = torch.randn(2, 8, 32)
    a_seq = torch.randn(2, 8, 8)
    rcwe_out = engine(x_seq, a_seq, plastic_runtime=True)

    candidates = torch.zeros(5, 8)
    candidates[3, 5:7] = 4.0

    z_curr = engine.codec.encode(x_seq)
    best_idx, scores = engine.forage_minimum_distinguishing_observation(z_curr, candidates)

    assert best_idx == 3, f"Expected high-leverage probe #3, got #{best_idx}"
    assert float(scores[best_idx].item()) > 1e-4, "Counterfactual variance split should be strictly non-zero!"

    pkt = codec.extract_diagnostic_packet(
        rcwe_output=rcwe_out,
        y_true=x_seq,
        candidate_actions=candidates,
        mdo_idx=best_idx,
        mdo_scores=scores,
        atp_vector=engine.metabolic_synapse.atp_budget,
    )
    advisory_text = codec.render_engineering_advisory(pkt)

    assert pkt.mdo_target_nodes == [5, 6]
    assert "v_5, v_6" in advisory_text
    assert f"Action #{best_idx}" in advisory_text
