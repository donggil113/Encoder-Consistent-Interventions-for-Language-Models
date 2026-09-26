"""Lexicon labels for the P3-REAL-02 behaviour task (standard library only).

The label is computed from generated text with a fixed word list, never from
the SAE. The wedding list is copied from Turner et al. (ActAdd,
arXiv:2308.10248, App. H), which scores wedding steering the same way.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

WORD_RE = re.compile(r"[a-z]+")

WEDDING_ACTADD = ("wedding", "weddings", "wed", "marry", "married", "marriage", "bride", "groom", "honeymoon")


def words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def count_hits(text: str, lexicon: Iterable[str]) -> int:
    lex = set(lexicon)
    return sum(1 for w in words(text) if w in lex)


def label(text: str, lexicon: Iterable[str]) -> Dict[str, float]:
    ws = words(text)
    lex = set(lexicon)
    hits = sum(1 for w in ws if w in lex)
    return {"hits": hits, "success": int(hits > 0), "rate": hits / len(ws) if ws else 0.0, "n_words": len(ws)}
