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
        ckpt_path = Path(checkpoint)
        tok_path = Path(tokenizer_path)
        if tok_path.is_dir():
            tok_path = tok_path / "tokenizer.json"
        self.tokenizer = load_tokenizer(tok_path)
        self.device = device
        # Use checkpoint's saved configuration and state
        if ckpt_path.suffix in (".yaml", ".yml"):
            config = load_config(ckpt_path)
            self.model = KothaGPT(config.model)
        elif ckpt_path.is_dir():
            config = load_config(ckpt_path / "config.json")
            self.model = KothaGPT(config.model)
            state_path = ckpt_path / "checkpoints" / "best.pt"
            if not state_path.is_file():
                state_path = ckpt_path / "checkpoints" / f"step-{0:07d}.pt"
                if not state_path.is_file():
                    raise FileNotFoundError(f"checkpoint not found in {ckpt_path}")
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            model_state = state.get("model_state", state) if isinstance(state, dict) else state
            self.model.load_state_dict(model_state, strict=True)
        else:
            config_path = ckpt_path.parent / "config.json"
            if not config_path.is_file():
                raise FileNotFoundError(f"config not found for checkpoint {ckpt_path}")
            config = load_config(config_path)
            self.model = KothaGPT(config.model)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model_state = state.get("model_state", state) if isinstance(state, dict) else state
            self.model.load_state_dict(model_state, strict=True)
        self.context_window = config.model.max_position_embeddings if "config" in locals() else 512
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
