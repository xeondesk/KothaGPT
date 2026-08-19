"""Token-efficiency benchmarks and the sample test texts.

The benchmark covers the Phase 1B test matrix: Bangla text, Bangla/English
mixed text, punctuation-heavy text, emoji, and code with Bangla comments.
"""

from __future__ import annotations

import statistics
from typing import Any

from .base import BaseTokenizer

__all__ = ["SAMPLE_TEXTS", "run_benchmark"]

SAMPLE_TEXTS: dict[str, str] = {
    "bangla": (
        "বাংলা ভাষা বাংলাদেশের মানুষের মাতৃভাষা। এটি একটি প্রাচীন এবং সমৃদ্ধ ভাষা, "
        "যার ইতিহাস হাজার বছরের পুরনো। রবীন্দ্রনাথ ঠাকুর, নজরুল ইসলাম এবং "
        "জীবনানন্দ দাশের মতো কবিরা এই ভাষায় তাদের অমর সৃষ্টি রেখে গেছেন। "
        "বাংলা সাহিত্য বিশ্বসাহিত্যের এক অনন্য সম্পদ। বাংলাদেশের স্বাধীনতা যুদ্ধের "
        "ইতিহাসে বাংলা ভাষা আন্দোলন একটি গুরুত্বপূর্ণ অধ্যায়।"
    ),
    "mixed": (
        "আমার ছোট্ট সোনামণি এবং আমি, একটি সুন্দর Sunday সকালে ঘুরতে গিয়েছিলাম। "
        "We visited the National Museum এবং তারপর lunch করলাম। The weather was "
        "beautiful এবং সবকিছু মনে হচ্ছিল perfect।"
    ),
    "punctuation": (
        "বাংলা ভাষা! এটি কি সত্যিই সুন্দর? হ্যাঁ, অবশ্যই! কমা, সেমিকোলন; কোলন: "
        "এবং ব্র্যাকেট (নিয়মিত) ব্যবহার করা হয়। প্রশ্নবোধক? বিস্ময়বোধক! "
        'উদ্ধৃতি "চিহ্ন" এবং ড্যাশ — সবই এখানে আছে।'
    ),
    "emoji": (
        "আমি বাংলা ভালোবাসি ❤️ এই ভাষা আমার গর্ব 🥰 বাংলাদেশ 🇧🇩 এ জন্মেছি আমরা 🎉 শুভ জন্মদিন 🎂 ধন্যবাদ 🙏 হাসি 😂"
    ),
    "code": (
        "def add(a, b):\n"
        "    # বাংলা মন্তব্য: দুটি সংখ্যা যোগ করুন\n"
        "    result = a + b\n"
        '    print(f"সমষ্টি: {result}")\n'
        "    return result"
    ),
}


def run_benchmark(tokenizer: BaseTokenizer, texts: dict[str, str] | None = None) -> dict[str, Any]:
    """Run the efficiency benchmark; returns per-set stats and the average."""
    if texts is None:
        texts = SAMPLE_TEXTS
    per_set: dict[str, dict[str, float]] = {}
    for name, text in texts.items():
        per_set[name] = tokenizer.stats(text)
    tpc = [s["tokens_per_char"] for s in per_set.values()]
    avg = statistics.mean(tpc) if tpc else 0.0
    return {"per_set": per_set, "avg_tokens_per_char": avg, "num_sets": len(tpc)}
