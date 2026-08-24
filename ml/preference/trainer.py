"""DPO stub wrapping SFT loss — preference alignment (WS-3)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ml.instruction.dataset import InstructionCollator
from ml.models import KothaGPT
from ml.preference.dataset import PreferenceRecord


def _encode_pair(tok, prompt: str, completion: str, max_len: int):
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion + "\n<eos>")
    ids = (p_ids + c_ids)[:max_len]
    prompt_len = min(len(p_ids), len(ids))
    labels = [-100]*prompt_len + ids[prompt_len:]
    return ids, labels


def run_dpo(
    model: KothaGPT,
    records: list[PreferenceRecord],
    tokenizer,
    *,
    device: str = "cpu",
    max_steps: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_length: int = 256,
    beta: float = 0.1,
):
    """Minimal DPO: chosen logprob - rejected logprob, no reference model (beta-scaled)."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps, total = 0, 0.0
    # Simple batching over records
    idx = 0
    while steps < max_steps:
        batch = records[idx: idx+batch_size]
        if not batch:
            idx = 0
            batch = records[idx: idx+batch_size]
        idx += batch_size
        # Build chosen vs rejected losses
        chosen_ids, chosen_labels, rejected_ids, rejected_labels = [], [], [], []
        max_len = max_length
        pad_id = getattr(tokenizer, "vocab", {}).get("<pad>", 3)
        for r in batch:
            c_ids, c_lab = _encode_pair(tokenizer, r.prompt, r.chosen, max_len)
            r_ids, r_lab = _encode_pair(tokenizer, r.prompt, r.rejected, max_len)
            chosen_ids.append(c_ids); chosen_labels.append(c_lab)
            rejected_ids.append(r_ids); rejected_labels.append(r_lab)
        # Pad
        def pad(seqs, pad_val):
            w = max(len(s) for s in seqs)
            return [s + [pad_val]*(w-len(s)) for s in seqs]
        # Compute DPO loss: -log(sigmoid(beta*(logp_chosen - logp_rejected)))
        # For stub, approximate logp as -CE loss
        def ce_loss(ids, labs):
            inp = torch.tensor(pad(ids, pad_id), dtype=torch.long, device=device)
            lab = torch.tensor(pad(labs, -100), dtype=torch.long, device=device)
            out = model(input_ids=inp, labels=lab)
            return out["loss"]
        loss_c = ce_loss(chosen_ids, chosen_labels)
        loss_r = ce_loss(rejected_ids, rejected_labels)
        # DPO: want chosen lower loss (higher logprob) => loss_c < loss_r
        dpo_loss = -torch.nn.functional.logsigmoid(beta * (loss_r - loss_c)).mean()
        opt.zero_grad(set_to_none=True)
        dpo_loss.backward()
        opt.step()
        total += float(dpo_loss.detach())
        steps += 1
    return {"steps": steps, "dpo_loss": total/max(steps, 1)}
