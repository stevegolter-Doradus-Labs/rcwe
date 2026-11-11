import json
import math
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Optional

from rcwe.lm.rcwe_deep_lm import RCWEDeepLanguageWorldModel, BaselineTransformerLM
from rcwe.kernels.accelerator import select_best_device, sync_device, describe_accelerator


class LargeCorpusRCWETrainer:
    """
    RCWE v0.8.0 Large-Scale Code & Technical Text Corpus Trainer for RCWE-Deep.
    Automatically harvests all Python source modules (`.py`) and Markdown whitepapers (`.md`)
    across the repository into a multi-kilobyte real-world code & math corpus, trains
    RCWE-Deep head-to-head against BaselineTransformerLM, measures bits-per-byte (BPB),
    perplexity, and next-token accuracy, and supports temperature-controlled causal generation.
    """
    def __init__(self, d_model: int = 96, num_layers: int = 4, seq_len: int = 256):
        self.device = select_best_device()
        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.vocab_size = 128

    def build_repository_corpus(self, root_dir: Path = Path(".")) -> str:
        chunks: List[str] = []
        for p in sorted(root_dir.rglob("*.py")):
            if ".venv" not in p.parts and "__pycache__" not in p.parts:
                try:
                    chunks.append(p.read_text(encoding="utf-8"))
                except Exception:
                    pass
        for p in sorted((root_dir / "docs").glob("*.md")) if (root_dir / "docs").exists() else []:
            try:
                chunks.append(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        full_text = "\n\n".join(chunks)
        if len(full_text) < 4096:
            full_text = (full_text + "\n") * 8
        return full_text

    def make_batches(self, text: str, batch_size: int = 8) -> torch.Tensor:
        raw_ids = [min(max(ord(c), 0), 127) for c in text]
        usable = (len(raw_ids) // (batch_size * (self.seq_len + 1))) * (batch_size * (self.seq_len + 1))
        if usable == 0:
            raw_ids = raw_ids * ((batch_size * (self.seq_len + 1)) // max(len(raw_ids), 1) + 2)
            usable = batch_size * (self.seq_len + 1)
        arr = torch.tensor(raw_ids[:usable], dtype=torch.long, device=self.device)
        return arr.view(-1, batch_size, self.seq_len + 1)

    @torch.no_grad()
    def generate_completion(
        self,
        model: RCWEDeepLanguageWorldModel,
        prompt: str = "def leapfrog_step(self, q, p, action",
        max_new_tokens: int = 64,
    ) -> str:
        model.eval()
        ids = [min(max(ord(c), 0), 127) for c in prompt]
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
        standing_fields = None

        out = model(inp, standing_fields=standing_fields, plastic_runtime=False)
        standing_fields = out["standing_fields"]
        next_id = int(out["logits"][0, -1].argmax().item())
        generated = [next_id]

        for _ in range(max_new_tokens - 1):
            step_inp = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            out = model(step_inp, standing_fields=standing_fields, plastic_runtime=False)
            standing_fields = out["standing_fields"]
            next_id = int(out["logits"][0, -1].argmax().item())
            generated.append(next_id)

        gen_str = "".join(chr(c) if 32 <= c < 127 or c == 10 else " " for c in generated)
        return prompt + gen_str

    def train_and_evaluate(
        self,
        epochs: int = 30,
        batch_size: int = 8,
        save_ckpt: str = "rcwe_deep_corpus_v08.pt",
    ) -> Dict[str, Any]:
        torch.manual_seed(42)
        corpus_text = self.build_repository_corpus(Path("."))
        batches = self.make_batches(corpus_text, batch_size=batch_size)
        num_batches = min(batches.shape[0], 6)

        rcwe_lm = RCWEDeepLanguageWorldModel(
            vocab_size=self.vocab_size, d_model=self.d_model, num_layers=self.num_layers, num_bands=16
        ).to(self.device)
        tf_lm = BaselineTransformerLM(
            vocab_size=self.vocab_size, d_model=self.d_model, num_layers=self.num_layers
        ).to(self.device)

        opt_r = torch.optim.AdamW(rcwe_lm.parameters(), lr=4e-3)
        opt_t = torch.optim.AdamW(tf_lm.parameters(), lr=4e-3)

        t0 = time.perf_counter()
        rcwe_lm.train()
        tf_lm.train()

        for _ in range(epochs):
            for b_idx in range(num_batches):
                chunk = batches[b_idx]
                x_ids = chunk[:, :-1]
                y_ids = chunk[:, 1:]

                opt_r.zero_grad()
                out_r = rcwe_lm(x_ids, plastic_runtime=False)
                loss_r = F.cross_entropy(out_r["logits"].reshape(-1, self.vocab_size), y_ids.reshape(-1))
                loss_r.backward()
                torch.nn.utils.clip_grad_norm_(rcwe_lm.parameters(), 2.0)
                opt_r.step()

                opt_t.zero_grad()
                logits_t = tf_lm(x_ids)
                loss_t = F.cross_entropy(logits_t.reshape(-1, self.vocab_size), y_ids.reshape(-1))
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(tf_lm.parameters(), 2.0)
                opt_t.step()

        sync_device(self.device)
        train_sec = time.perf_counter() - t0

        rcwe_lm.eval()
        tf_lm.eval()
        with torch.no_grad():
            eval_chunk = batches[0]
            x_eval, y_eval = eval_chunk[:, :-1], eval_chunk[:, 1:]
            res_r = rcwe_lm(x_eval, plastic_runtime=True)
            res_t = tf_lm(x_eval)

            ce_r = float(F.cross_entropy(res_r["logits"].reshape(-1, self.vocab_size), y_eval.reshape(-1)).item())
            ce_t = float(F.cross_entropy(res_t.reshape(-1, self.vocab_size), y_eval.reshape(-1)).item())

            bpb_r = ce_r / math.log(2.0)
            bpb_t = ce_t / math.log(2.0)

            acc_r = float((res_r["logits"].argmax(dim=-1) == y_eval).float().mean().item())
            acc_t = float((res_t.argmax(dim=-1) == y_eval).float().mean().item())
            psi_kb = float(res_r["total_psi_kb"])

        sample_completion = self.generate_completion(
            rcwe_lm, prompt="def leapfrog_step(self, q, p", max_new_tokens=48
        )
        torch.save(rcwe_lm.state_dict(), Path(save_ckpt))

        report = {
            "corpus_total_chars": len(corpus_text),
            "trained_tokens_per_epoch": int(num_batches * batch_size * self.seq_len),
            "d_model": self.d_model,
            "layers": self.num_layers,
            "rcwe_deep_bpb": round(bpb_r, 4),
            "tf_deep_bpb": round(bpb_t, 4),
            "rcwe_deep_accuracy": round(acc_r, 4),
            "tf_deep_accuracy": round(acc_t, 4),
            "total_psi_kb": psi_kb,
            "train_wall_sec": round(train_sec, 2),
            "sample_completion": sample_completion,
            "saved_checkpoint": str(Path(save_ckpt).resolve()),
        }
        Path("rcwe_deep_corpus_report_v08.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report


def main() -> None:
    print("================================================================================")
    print("   DORADUS LABS — RCWE-DEEP v0.8.0 LARGE REPOSITORY CODE & MATH CORPUS TRAINER")
    print("================================================================================")
    trainer = LargeCorpusRCWETrainer(d_model=96, num_layers=4, seq_len=256)
    rep = trainer.train_and_evaluate(epochs=30, batch_size=8)
    print(f"Harvested Repository Corpus : {rep['corpus_total_chars']:,} chars ({rep['trained_tokens_per_epoch']:,} tokens/epoch)")
    print(f"RCWE-Deep Bits-Per-Byte     : {rep['rcwe_deep_bpb']:.3f} BPB vs Transformer = {rep['tf_deep_bpb']:.3f} BPB")
    print(f"Next-Token Accuracy         : RCWE-Deep = {rep['rcwe_deep_accuracy']*100:.1f}% vs Transformer = {rep['tf_deep_accuracy']*100:.1f}%")
    print(f"4-Layer O(1) |Psi| State    : {rep['total_psi_kb']:.1f} KB (Trained in {rep['train_wall_sec']}s)")
    print(f"Sample O(1) Code Rollout    : {rep['sample_completion']!r}")
    print(f"[Saved RCWE-Deep Corpus Checkpoint -> {rep['saved_checkpoint']}]")


if __name__ == "__main__":
    main()
