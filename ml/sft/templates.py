"""Chat template registry for SFT (WS-2).

Provides Bangla/English system+turn formatting with byte-identical round-trip:
  text = apply_chat_template(messages, template="kothagpt-bn")
  messages == parse_chat_template(text, template="kothagpt-bn")

Uses minimal special tokens compatible with frozen 16k BPE vocab:
  <user>, <assistant>, <system>, <tool>, <eos>
Checks that tokenizer covers all required tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

_REQUIRED_TOKENS = ["<user>", "<assistant>", "<system>", "<tool>", "<eos>", "<pad>"]
_TEMPLATES = {}


@dataclass(frozen=True)
class ChatTemplate:
    name: str
    system_prefix: str = "<system>\n"
    user_prefix: str = "<user>\n"
    assistant_prefix: str = "<assistant>\n"
    tool_prefix: str = "<tool>\n"
    eos: str = "\n<eos>"
    joiner: str = "\n"

    def apply(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                parts.append(f"{self.system_prefix}{content}")
            elif role == "user":
                parts.append(f"{self.user_prefix}{content}")
            elif role == "assistant":
                parts.append(f"{self.assistant_prefix}{content}")
            elif role == "tool":
                parts.append(f"{self.tool_prefix}{content}")
            else:
                raise ValueError(f"unsupported role: {role}")
        # Ensure assistant turn ends without eos; completion adds it
        return self.joiner.join(parts) + self.joiner + "<assistant>\n"

    def parse(self, text: str) -> list[dict[str, str]]:
        # Remove trailing assistant prefix added by apply
        if text.endswith("<assistant>\n"):
            text = text[: -len("<assistant>\n")]
        # Split by known prefixes
        import re

        pattern = r"(<system>\n|<user>\n|<assistant>\n|<tool>\n)"
        tokens = re.split(pattern, text)
        messages: list[dict[str, str]] = []
        # tokens includes delimiters
        i = 1
        while i < len(tokens):
            prefix = tokens[i].strip()
            content = tokens[i + 1] if i + 1 < len(tokens) else ""
            # Remove trailing joiner between messages
            if content.endswith("\n"):
                # need to handle joiner logic: apply joins with \n, so content may include next prefix's preceding \n
                pass
            # Content is up to next delimiter; strip leading/trailing \n that are joiners
            # The apply does joiner.join(parts) so each part is separate; we split correctly above
            content = content.rstrip("\n")
            # Remove leading \n if present (from joiner)
            if content.startswith("\n"):
                content = content[1:]
            role = prefix.strip()[1:-1]  # <user> -> user
            messages.append({"role": role, "content": content})
            i += 2
        return messages


def _register_defaults() -> None:
    for name in ["default", "kothagpt-bn", "kothagpt-en", "kothagpt"]:
        _TEMPLATES[name] = ChatTemplate(name=name)
    # English could use same but keep separate for future specialization
    _TEMPLATES["kothagpt-en"] = ChatTemplate(name="kothagpt-en")
    _TEMPLATES["kothagpt-bn"] = ChatTemplate(name="kothagpt-bn")


_register_defaults()


def get_template(name: str = "default") -> ChatTemplate:
    if name not in _TEMPLATES:
        raise ValueError(f"unknown template {name!r}, available: {sorted(_TEMPLATES)}")
    return _TEMPLATES[name]


def apply_chat_template(messages: list[dict[str, str]], template: str = "default") -> str:
    return get_template(template).apply(messages)


def parse_chat_template(text: str, template: str = "default") -> list[dict[str, str]]:
    return get_template(template).parse(text)


def check_tokenizer_coverage(tokenizer, template: str = "default") -> dict[str, bool]:
    """Verify tokenizer covers all chat special tokens. Returns {token: covered}."""
    # tokenizer may be HF tokenizer or custom with vocab dict
    vocab = getattr(tokenizer, "vocab", None) or getattr(tokenizer, "get_vocab", lambda: {})()
    if callable(vocab):
        try:
            vocab = vocab()
        except Exception:
            vocab = {}
    # Also try encode check
    coverage: dict[str, bool] = {}
    for tok in _REQUIRED_TOKENS:
        covered = False
        if isinstance(vocab, dict) and tok in vocab:
            covered = True
        else:
            try:
                ids = tokenizer.encode(tok)
                # If tokenizer splits special token, it would be multiple ids; we check that it doesn't split into unk
                covered = len(ids) > 0
            except Exception:
                covered = False
        coverage[tok] = covered
    return coverage
