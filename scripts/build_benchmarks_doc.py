#!/usr/bin/env python3
"""
Run every benchmark suite with the current code on this machine and write docs/BENCHMARKS.md.

    python scripts/build_benchmarks_doc.py            (or: make benchmarks)

The page it writes states exactly what was measured, on what hardware, and how each number
should be read. Nothing in it is copied from an earlier run.
"""
from __future__ import annotations

import contextlib
import io
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rcwe  # noqa: E402
from rcwe.kernels.accelerator import describe_accelerator, select_best_device  # noqa: E402


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def ratio(ours: float, theirs: float) -> str:
    if ours <= 0 or theirs <= 0:
        return "n/a"
    r = theirs / ours
    return f"{r:.1f}x lower" if r >= 1 else f"{1 / r:.1f}x higher"


@torch.no_grad()
def leak_free_mutation(bench) -> dict:
    """The validation suite's topology-mutation stream, scored before the engine sees each answer."""
    bench.rcwe.eval()
    bench.baseline_tf.eval()
    bench.rcwe.reset_runtime_adaptation()
    obs = bench.sandbox.reset(batch_size=4, worldline_regime=1)
    field = None
    disturbance = torch.zeros(4, 1, 32, device=bench.device)
    disturbance[:, :, :16] = 0.40
    rcwe_err, tf_err = [], []
    for _ in range(20):
        act = torch.randn(4, 1, 8, device=bench.device) * 0.3
        x = obs.unsqueeze(1)
        nxt, _ = bench.sandbox.step(act.squeeze(1))
        y = nxt.unsqueeze(1) + disturbance
        prior = bench.rcwe(x, act, standing_field=field, plastic_runtime=False)           # prediction: no access to y
        rcwe_err.append(float(F.mse_loss(prior["output"], y)))
        tf_err.append(float(F.mse_loss(bench.baseline_tf(x, act), y)))
        field = bench.rcwe(x, act, standing_field=field, plastic_runtime=True, y_target=y)["standing_field"]   # then learn
        obs = nxt
    bench.rcwe.reset_runtime_adaptation()
    return {"rcwe": sum(rcwe_err[-5:]) / 5.0, "tf": sum(tf_err[-5:]) / 5.0}


def toolkit_table(seeds) -> list:
    from rcwe.toolkit import TelemetryModel
    from rcwe.toolkit.demo import make_pump_station

    rows = []
    for seed in seeds:
        demo = make_pump_station(rows=4200, faults=True, fault_start=3000, seed=seed)
        normal, incident = demo.table.slice(0, 3000), demo.table.slice(3000, None)
        truth = [dict(e, start=e["start"] - 3000, end=e["end"] - 3000) for e in demo.events]
        clean = make_pump_station(rows=4200, faults=False, seed=seed).table.slice(3000, None)
        started = time.perf_counter()
        model = TelemetryModel(seed=seed).fit(normal)
        fit_seconds = time.perf_counter() - started
        result = model.detect(incident)
        blamed = 0
        for fault in truth:
            hits = [e for e in result.events if not (e.end < fault["start"] - 3 or e.start > fault["end"] + 3)]
            blamed += bool(hits and hits[0].top_sensor in fault["sensors"])
        card = model.fit_summary_
        rows.append({
            "seed": seed, "fit_s": fit_seconds, "skill": card.skill, "fc48": card.forecast_skill.get("48", float("nan")),
            "caught": result.metrics["incidents_detected"], "false": result.metrics["false_alarm_events"], "blamed": blamed,
            "auroc": result.metrics["auroc"], "f1": result.metrics["point_f1"], "clean_events": model.detect(clean).n_events,
        })
    return rows


# Where `real_world_suite.generate_benchmark_csvs` plants its faults (the sensors it adds an offset to).
INJECTED = {
    "ETT_Transformer_Thermal": ("q_2", "q_3"),
    "SWaT_CyberPhysical_Water": ("q_4", "q_5"),
    "NASA_Turbofan_Bearing": ("q_7", "q_8"),
}


def toolkit_on_regime_shift_logs() -> dict:
    """Ask the toolkit which sensor each regime-shift log's fault started on."""
    from rcwe.benchmarks.real_world_suite import RealWorldBenchmarkSuite
    from rcwe.paths import output_path
    from rcwe.toolkit import TelemetryModel, load_table

    found = {}
    for domain, path in RealWorldBenchmarkSuite.generate_benchmark_csvs(Path(output_path("benchmarks_data"))).items():
        table = load_table(path)
        result = TelemetryModel(seed=0).fit(table.slice(0, 20)).detect(table)
        event = result.events[0] if result.events else None
        named = [s["name"] for s in event.sensors] if event else []
        found[domain] = {
            "injected": INJECTED[domain],
            "lead_is_injected": bool(named and named[0] in INJECTED[domain]),
            "text": f"starts at step {event.start}; names {', '.join(named)}" if event else "nothing detected",
        }
    return found


def controls_table(seeds) -> list:
    """
    A log in which an operator moves the valve and the pump speed by hand, with no faults.
    The same model is trained twice: with the controls left as they are by default (they
    might be reacting to the sensors), and with both declared as commands.
    """
    import numpy as np

    from rcwe.toolkit import TelemetryModel
    from rcwe.toolkit.demo import make_pump_station

    rows = []
    for seed in seeds:
        demo = make_pump_station(rows=4200, faults=False, seed=seed, operator_steps=18)
        train, test = demo.table.slice(0, 3000), demo.table.slice(3000, None)
        moved = sum(1 for move in demo.operator_steps if move["start"] >= 3000 or move["end"] >= 3000)
        last = train.to_rows()[-1]
        hold = np.tile([last["act_valve_open"], last["act_pump_speed"]], (24, 1))
        opened = hold.copy()
        opened[:, 0] += 0.10
        plant = last["flow_m3h"] * 0.10 / (0.4 + last["act_valve_open"])      # what the simulated station does
        for mode, commands in (("default", None), ("commands", True)):
            model = TelemetryModel(seed=seed).fit(train, commands=commands)
            result = model.detect(test)
            flow = model.sensors.index("flow_m3h")
            history = train.slice(2800, None)
            a, b = model.forecast(24, history, future_controls=hold), model.forecast(24, history, future_controls=opened)
            rows.append({
                "seed": seed, "mode": mode, "skill": model.fit_summary_.skill, "steps_in_test": moved,
                "events": result.n_events, "pointed": sum(1 for e in result.events if e.control_changes),
                "worst": max((e.severity for e in result.events), key=["low", "medium", "high", "critical"].index, default="none"),
                "response": float(b.values[-1, flow] - a.values[-1, flow]), "plant": plant,
            })
    return rows


def main() -> None:
    from rcwe.benchmarks.real_world_suite import RealWorldBenchmarkSuite
    from rcwe.control.self_healing import AutonomousSelfHealingAgent
    from rcwe.diagnostics.extreme_bench import run_extreme_context_sweep
    from rcwe.diagnostics.validation import ScientificValidationBenchmark
    from rcwe.kernels.metal_msl import benchmark_msl_131k_scaling
    from rcwe.lm.corpus_trainer import LargeCorpusRCWETrainer
    from rcwe.lm.rcwe_deep_lm import run_deep_lm_benchmark
    from rcwe.multimodal.spectral_fusion import run_multimodal_substation_benchmark
    from rcwe.rl.symplectic_rl import run_symplectic_rl_benchmark
    from rcwe.swarm.wavefield_swarm import run_distributed_swarm_benchmark
    from rcwe.symbolic.algebra_task import run_symbolic_reasoning_benchmark
    from rcwe.symbolic.code_reasoning import run_code_reasoning_benchmark

    device = select_best_device()
    started = time.perf_counter()
    out = []
    say = out.append

    say("# Benchmarks")
    say("")
    say(f"Generated by `python scripts/build_benchmarks_doc.py` (`make benchmarks`) with **RCWE {rcwe.__version__}**.")
    say("")
    say("| | |")
    say("|---|---|")
    say(f"| Machine | {platform.system()} {platform.machine()}, {torch.get_num_threads()} CPU threads |")
    say(f"| Accelerator | {describe_accelerator(device)} |")
    say(f"| Python / PyTorch | {platform.python_version()} / {torch.__version__} |")
    say("")
    say("Every figure below was measured by this run. Results move with hardware, PyTorch version and random seed, "
        "so regenerate the page on your own machine before quoting it. The author's earlier Apple Silicon runs are "
        "kept in [`reference_results/`](reference_results/).")
    say("")

    # ------------------------------------------------------------------ toolkit
    print("toolkit demo ...", flush=True)
    rows = toolkit_table([7, 1, 2, 3, 4])
    say("## 1. Toolkit: anomaly detection and forecasting on the demo data")
    say("")
    say("`TelemetryModel` trained on 3,000 rows of healthy pump-station telemetry, then asked to scan the following "
        "1,200 rows containing five planted faults (drift, level shift, spike, stuck sensor, correlation break). "
        "Five independently generated datasets. A fault counts as caught if a detected event overlaps it; it counts as "
        "correctly attributed if the first sensor the event names is one that was actually faulted.")
    say("")
    say("| Seed | Faults caught | Right sensor blamed | False alarms | False alarms on a clean log | AUROC | Point F1 | One-step skill vs naive | 48-step forecast skill | Training time |")
    say("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        say(f"| {r['seed']} | {r['caught']} / 5 | {r['blamed']} / 5 | {r['false']} | {r['clean_events']} | {r['auroc']:.3f} | {r['f1']:.2f} | "
            f"{r['skill'] * 100:+.0f}% | {r['fc48'] * 100:+.0f}% | {r['fit_s']:.0f} s |")
    say(f"| **Total** | **{sum(r['caught'] for r in rows)} / {5 * len(rows)}** | **{sum(r['blamed'] for r in rows)} / {5 * len(rows)}** | "
        f"**{sum(r['false'] for r in rows)}** | **{sum(r['clean_events'] for r in rows)}** | | | | | |")
    say("")
    say("*Skill* is the reduction in mean squared error against the naive forecast \"the value stays where it is\", "
        "measured on rows held out from training. The demo data is synthetic; expect lower numbers on noisy real logs, "
        "and read the skill figure in your own report before relying on forecasts.")
    say("")

    # ------------------------------------------------------------------ controls
    print("controls as commands ...", flush=True)
    ctl = controls_table([7, 1, 2])
    say("## 2. Toolkit: control inputs, by default and declared as commands")
    say("")
    say("The same pump station with no faults, but with an operator who moves the valve and the pump speed by hand "
        "(18 step changes per control over 4,200 rows; the plant responds as it should). The model is trained on the first "
        "3,000 rows twice: with the controls treated as they are by default, where a control is only used up to the row "
        "being predicted *from* because it might be reacting to the sensors, and with both declared as commands "
        "(`fit(..., commands=True)` or `--commands all`), where the model is also told their value for the row it predicts.")
    say("")
    say("| Seed | Controls | One-step skill | Events on 1,200 fault-free rows with operator steps | ...of which point at the control that moved | "
        "Forecast response to a planned +0.10 valve step (24 steps) | What the plant does |")
    say("|---:|---|---:|---:|---:|---:|---:|")
    for r in ctl:
        say(f"| {r['seed']} | {r['mode']} | {r['skill'] * 100:+.0f}% | {r['events']}" + (f" (worst: {r['worst']})" if r['events'] else "")
            + f" | {r['pointed']} | {r['response']:+.2f} m3/h | {r['plant']:+.2f} m3/h |")
    say("")

    def share(mode: str) -> float:
        picked = [r for r in ctl if r["mode"] == mode]
        return sum(r["response"] for r in picked) / sum(r["plant"] for r in picked)

    say(f"- Told nothing about the row it predicts, the model cannot anticipate an operator's step: its forecast moves by "
        f"{share('default') * 100:.0f}% of what the plant does. With the controls declared as commands it moves by "
        f"{share('commands') * 100:.0f}%: the right direction and most, not all, of the size.")
    def total(mode: str, key: str) -> int:
        return sum(r[key] for r in ctl if r["mode"] == mode)

    say(f"- An operator's step is not a fault. In the scanned rows the operator made {total('default', 'steps_in_test')} step "
        f"changes in all. They raised {total('default', 'events')} event(s) with the controls as by default and "
        f"{total('commands', 'events')} with them declared as commands; {total('default', 'pointed')} and "
        f"{total('commands', 'pointed')} of those say which control changed as they began, so they can be told from a fault "
        "at a glance. (The training log contains steps of the same kind, which is what keeps most of them quiet.)")
    say("- Declare as commands only controls that are set independently of the sensors (setpoints, schedules). A controller "
        "output reacts to the sensors within the same sample; told its value, the model would learn the reaction and explain "
        "faults away. That is why it is not the default.")
    say("")

    # ------------------------------------------------------------------ validation
    print("multi-seed validation ...", flush=True)
    bench = ScientificValidationBenchmark(seed=42)
    report = quiet(bench.run_multiseed_validation, seeds=[42, 101, 202], epochs=55)
    ms = report["multiseed_statistics"]
    leak_free = []
    for seed in (42, 101, 202):
        bench._init_models(seed)
        quiet(bench.train_and_calibrate, epochs=55)
        leak_free.append(leak_free_mutation(bench))
    lf_r = sum(v["rcwe"] for v in leak_free) / len(leak_free)
    lf_t = sum(v["tf"] for v in leak_free) / len(leak_free)
    env = report["environment"]
    say("## 3. Engine vs frozen Transformer on the flow sandbox")
    say("")
    say(f"Three seeds, {env['rcwe_params']:,} engine parameters vs {env['baseline_tf_params']:,} in the Transformer baseline. "
        "After training, the sandbox topology is mutated and an offset is added to the observations; the engine keeps "
        "adapting online (no backpropagation) while the Transformer's weights stay frozen.")
    say("")
    say("| Metric | RCWE | Frozen Transformer | Difference |")
    say("|---|---:|---:|---:|")
    r, t = ms["rcwe_post_mutation_mse"], ms["tf_post_mutation_mse"]
    say(f"| Post-mutation MSE, original protocol | {r['mean']:.5f} ± {r['std']:.5f} | {t['mean']:.5f} ± {t['std']:.5f} | {ratio(r['mean'], t['mean'])} |")
    say(f"| Post-mutation MSE, scored before the engine sees the answer | {lf_r:.5f} | {lf_t:.5f} | {ratio(lf_r, lf_t)} |")
    r, t = ms["rcwe_hamiltonian_drift"], ms["tf_hamiltonian_drift"]
    say(f"| Energy (Hamiltonian) drift of predictions | {r['mean']:.5f} ± {r['std']:.5f} | {t['mean']:.5f} ± {t['std']:.5f} | {ratio(r['mean'], t['mean'])} |")
    acc = ms["regime_id_accuracy"]
    say(f"| Regime identification accuracy (3 regimes, fresh 8-step rollouts) | {acc['mean'] * 100:.1f}% ± {acc['std'] * 100:.1f}% | n/a | |")
    say("")
    say("How to read this:")
    say("")
    say("- The comparison measures the value of **adapting online** against a model that cannot. It is not a claim that "
        "the architecture out-predicts a Transformer that is also allowed to adapt.")
    agree = abs(lf_r - ms["rcwe_post_mutation_mse"]["mean"]) <= 0.05 * ms["rcwe_post_mutation_mse"]["mean"]
    say("- In the original protocol the engine's `forward` is given the true next state (`y_target`) for the step being "
        "scored; it uses it to weight its world-lines before producing the prediction, which the baseline cannot do. The "
        "second row scores the prediction first and lets the engine adapt afterwards. "
        + ("Here the two rows agree, so that access is not what drives the result. " if agree else
           "Here the two rows differ, so part of the original figure comes from that access. ")
        + "The toolkit (`TelemetryModel`) and Mission Control always work the second way.")
    say("- Regime identification is measured on short rollouts started from a fresh sandbox state. In a long continuous "
        "stream (Mission Control) single-step evidence is noisier and the dashboard smooths it over time.")
    say("")

    # ------------------------------------------------------------------ scaling
    print("context scaling ...", flush=True)
    sweep = quiet(run_extreme_context_sweep)
    say("## 4. Context-length scaling")
    say("")
    say("One forward pass over sequences of growing length (batch 2, 64-dimensional model).")
    say("")
    say("| Sequence length | Transformer latency | RCWE latency | Transformer attention workspace | RCWE standing wavefield |")
    say("|---:|---:|---:|---:|---:|")
    for row in sweep:
        tf_cell = f"{row['tf_latency_ms']:.1f} ms" if row["tf_latency_ms"] is not None else "not run (over budget)"
        say(f"| {row['seq_len']:,} | {tf_cell} | {row['rcwe_latency_ms']:.1f} ms | {row['tf_workspace_kb'] / 1024:,.0f} MB | {row['measured_psi_kb']:.1f} KB |")
    say("")
    say("- The wavefield column is measured from the tensor the engine carries between calls: its size does not depend on sequence length.")
    say("- The attention workspace is computed from the sequence length, not measured. Above a 1 GB budget the baseline "
        "is **not run**, so no out-of-memory failure was observed here: the table shows where a dense-attention model "
        "would need that much memory, and that the engine still runs at those lengths.")
    say("")

    # ------------------------------------------------------------------ real-world style suite
    print("regime-shift suite ...", flush=True)
    rw = quiet(RealWorldBenchmarkSuite.run_all_benchmarks)
    say("## 5. Regime-shift suite (ETT / SWaT / NASA-style)")
    say("")
    say("Three **synthetic** 40-step logs shaped like well-known industrial datasets (they are generated by "
        "`real_world_suite.py`, not downloaded), each with a fault injected at step 20. Same original protocol as section 3.")
    say("")
    attribution = toolkit_on_regime_shift_logs()
    say("| Domain | Sensors / controls | RCWE post-shift MSE | Frozen Transformer | Difference | Fault injected on | Toolkit: first event |")
    say("|---|---:|---:|---:|---:|---|---|")
    for b in rw["benchmarks"]:
        found = attribution[b["domain"]]
        say(f"| {b['domain'].replace('_', ' ')} | {b['sensors']} / {b['controls']} | {b['rcwe_post_shift_mse']:.5f} | {b['tf_post_shift_mse']:.5f} | "
            f"{ratio(b['rcwe_post_shift_mse'], b['tf_post_shift_mse'])} | {', '.join(found['injected'])} at step 20 | {found['text']} |")
    say("")
    right = sum(found["lead_is_injected"] for found in attribution.values())
    say(f"The last column is a separate check with the toolkit: `TelemetryModel` is trained on the 20 fault-free rows of each "
        f"log and then scans all 40. The sensor an event names first was one of the injected ones in **{right} of "
        f"{len(attribution)}** logs. Twenty rows are far too few to calibrate scores, and the model says so; read this as a "
        "check of attribution, not of sensitivity.")
    say("")

    # ------------------------------------------------------------------ symbolic + code
    print("symbolic and code tasks ...", flush=True)
    sym = quiet(run_symbolic_reasoning_benchmark, seed=42, epochs=45)
    code = quiet(run_code_reasoning_benchmark, seed=42, epochs=35)
    say("## 6. Symbolic conservation and code-trace tasks")
    say("")
    say("| Task | RCWE | Transformer | Difference |")
    say("|---|---:|---:|---:|")
    say(f"| Conservation-sum error on a 16-register ledger | {sym['rcwe_symbolic_conservation_error']:.5f} | {sym['tf_symbolic_conservation_error']:.5f} | "
        f"{ratio(sym['rcwe_symbolic_conservation_error'], sym['tf_symbolic_conservation_error'])} |")
    say(f"| MSE after a mid-stream rule change (RCWE adapts online) | {sym['rcwe_rule_mutation_mse']:.5f} | {sym['tf_rule_mutation_mse']:.5f} | "
        f"{ratio(sym['rcwe_rule_mutation_mse'], sym['tf_rule_mutation_mse'])} |")
    say(f"| Code-trace prediction MSE | {code['rcwe_code_trace_mse']:.5f} | {code['tf_code_trace_mse']:.5f} | "
        f"{ratio(code['rcwe_code_trace_mse'], code['tf_code_trace_mse'])} |")
    say("")
    say("The code-trace result is within noise of the baseline and can fall on either side depending on the platform.")
    say("")

    # ------------------------------------------------------------------ phase 2
    print("phase-2 tracks ...", flush=True)
    msl = quiet(benchmark_msl_131k_scaling, d_model=64, batch_size=1)
    heal = quiet(AutonomousSelfHealingAgent(seed=42).run_closed_loop_healing_episode)
    lm = quiet(run_deep_lm_benchmark, seed=42, epochs=45)
    corpus = quiet(LargeCorpusRCWETrainer(d_model=96, num_layers=4, seq_len=256).train_and_evaluate, epochs=30, batch_size=8)
    say("## 7. Long-context kernel, self-healing control, language model")
    say("")
    say(f"Holographic scan alone (custom Metal shader compiled: **{'yes' if msl['msl_shader_compiled'] else 'no, PyTorch fallback'}**):")
    say("")
    say("| Sequence length | Scan latency | Wavefield size |")
    say("|---:|---:|---:|")
    for row in msl["sweeps"]:
        say(f"| {row['seq_len']:,} | {row['rcwe_msl_latency_ms']:.1f} ms | {row['measured_psi_kb']:.1f} KB |")
    say("")
    say("| Track | Result |")
    say("|---|---|")
    say(f"| Closed-loop self-healing of a sinkhole fault | residual deviation {heal['self_healed_residual_deviation']:.4f} with the controller vs "
        f"{heal['unhealed_sinkhole_energy_loss']:.4f} without ({heal['stabilization_gain_ratio']:.1f}x); regime identified as W{heal['identified_regime']} |")
    say(f"| RCWE-Deep on a short repeated code snippet ({lm['sequence_tokens']} tokens) | next-token accuracy {lm['rcwe_deep_token_accuracy'] * 100:.1f}% vs "
        f"{lm['tf_deep_token_accuracy'] * 100:.1f}% for a Transformer; measured on the **training** sequence |")
    say(f"| RCWE-Deep on the repository source ({corpus['corpus_total_chars']:,} characters) | {corpus['rcwe_deep_bpb']:.2f} bits/byte vs "
        f"{corpus['tf_deep_bpb']:.2f} for a Transformer after {corpus['train_wall_sec']:.0f} s of training; measured on a **training** batch |")
    say("")
    say("The self-healing controller's feedback gains and the faulted nodes are written into the benchmark; the engine's "
        "part is identifying the regime. The two language-model rows report fit to data the models were trained on, so "
        "they show capacity to memorise under equal training time, not generalisation.")
    say("")

    # ------------------------------------------------------------------ v0.9 tracks
    print("v0.9 tracks ...", flush=True)
    mm = quiet(run_multimodal_substation_benchmark, epochs=30)
    rl = quiet(run_symplectic_rl_benchmark, steps=40)
    sw = quiet(run_distributed_swarm_benchmark, steps=30)
    say("## 8. Multimodal fusion, symplectic RL, wavefield swarm")
    say("")
    say("| Track | RCWE | Comparison | Difference |")
    say("|---|---:|---:|---:|")
    say(f"| Thermal + telemetry fusion: MSE after a coupled fault (online adaptation) | {mm['rcwe_multimodal_mse']:.5f} | {mm['tf_multimodal_mse']:.5f} (frozen Transformer) | "
        f"{ratio(mm['rcwe_multimodal_mse'], mm['tf_multimodal_mse'])} |")
    say(f"| RL plant deviation under a persistent disturbance | {rl['rcwe_rl_post_mutation_deviation']:.5f} | {rl['frozen_policy_post_mutation_deviation']:.5f} (frozen PD policy) | "
        f"{ratio(rl['rcwe_rl_post_mutation_deviation'], rl['frozen_policy_post_mutation_deviation'])} |")
    say(f"| Swarm: error on downstream agents at the moment a cascade reaches them | {sw['swarm_coupled_cascade_mse']:.5f} | {sw['isolated_agents_cascade_mse']:.5f} (no gossip) | "
        f"{ratio(sw['swarm_coupled_cascade_mse'], sw['isolated_agents_cascade_mse'])} |")
    say("")
    say("- The RL agent's plastic update is driven by a corrective-force signal supplied by the benchmark. Once the "
        "disturbance stops, the adapted policy holds the plant less tightly than the frozen one; Mission Control shows both phases.")
    say("- The swarm figure covers only the two steps at first impact. Gossip works by pre-loading neighbours with the "
        "fault before it reaches them, so in the steps before impact those neighbours mispredict. Over a whole incident "
        "the coupled swarm's mean error can be higher than the isolated one's; Mission Control reports that cost next to the gain.")
    say("")
    say(f"_Total run time: {time.perf_counter() - started:.0f} s._")

    target = ROOT / "docs" / "BENCHMARKS.md"
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
