"""WS-1 inference engine stub — torch-backed generation behind Backend seam."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

try:
    import torch
    from ml.models import KothaGPT, load_config
    from ml.tokenizer import load_tokenizer
except ImportError:
    torch = None  # type: ignore

class KothaGPTEngine:
    def __init__(self, checkpoint: str | Path, tokenizer_path: str | Path, device: str = "cpu"):
        if torch is None:
            raise ImportError("torch required for inference engine")
        cfg = load_config(checkpoint) if str(checkpoint).endswith(".yaml") else None
        # fallback: load from checkpoint dir
        tok_path = Path(tokenizer_path)
        if tok_path.is_dir():
            tok_path = tok_path / "tokenizer.json"
        self.tokenizer = load_tokenizer(tok_path)
        # For stub, init small model; real engine loads state_dict from checkpoint
        from ml.models.config import ModelConfig
        mcfg = ModelConfig(vocab_size=len(self.tokenizer.vocab), hidden_size=128, num_layers=2, num_heads=4, max_position_embeddings=512)
        self.model = KothaGPT(mcfg)
        self.device = device
        self.model.to(device).eval()

    def generate(self, prompt: str, max_new_tokens: int = 32) -> Iterator[str]:
        ids = self.tokenizer.encode(prompt)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                inp = torch.tensor([ids], device=self.device)
                out = self.model(inp)
                logits = out["logits"] if isinstance(out, dict) else out.logits
                nxt = int(logits[0, -1].argmax().item())
                if nxt == self.tokenizer.vocab.get("<eos>", 2):
                    break
                ids.append(nxt)
                yield self.tokenizer.id_to_token.get(nxt, "<unk>")

