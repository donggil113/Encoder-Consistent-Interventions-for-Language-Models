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


# ---------------------------------------------------------------- bounded matching (P3-REAL-02-SPACE)
# Rules fixed before any corpus count (configs/p3_real02_space.json, task.matching):
#  * lowercase; a word is a maximal run of ASCII letters [a-z]; everything else is a boundary
#    (so "Earth's" -> "earth", "s"; "space-time" -> "space", "time");
#  * a lexicon entry is a sequence of such words and matches whole consecutive words only
#    (no substrings: "spacecrafts" does not match "spacecraft"; no stemming or inflection);
#  * a word cut by a text-span edge is a fragment and never matches: the first word of a span
#    is dropped when the span starts with a letter and the preceding text ends with a letter,
#    and the last word is dropped when the span ends with a letter and the following text
#    starts with a letter (BPE window/prompt boundaries can split words).

LETTER_RE = re.compile(r"[A-Za-z]")


def _entries(lexicon: Iterable[str]) -> List[tuple]:
    return sorted({tuple(words(e)) for e in lexicon if words(e)}, key=len, reverse=True)


def bounded_hits(text: str, lexicon: Iterable[str], before: str = "", after: str = "") -> int:
    """Whole-word lexicon matches in ``text``; ``before``/``after`` is the adjacent text outside
    the span (used only to detect edge fragments)."""
    ws = [m.group() for m in WORD_RE.finditer(text.lower())]
    if ws and text[:1] and LETTER_RE.match(text[0]) and before[-1:] and LETTER_RE.match(before[-1]):
        ws = ws[1:]
    if ws and text[-1:] and LETTER_RE.match(text[-1]) and after[:1] and LETTER_RE.match(after[0]):
        ws = ws[:-1]
    hits, i = 0, 0
    ents = _entries(lexicon)
    while i < len(ws):
        for e in ents:
            if tuple(ws[i:i + len(e)]) == e:
                hits += 1
                i += len(e)
                break
        else:
            i += 1
    return hits


def bounded_label(text: str, lexicon: Iterable[str], before: str = "", n_tokens: int = 0) -> Dict[str, float]:
    """Label of a generated continuation: edge fragment at the start is dropped (``before`` is the
    prompt's last token text); the final word is counted as generated."""
    hits = bounded_hits(text, lexicon, before=before)
    n_words = len(words(text))
    return {"hits": hits, "success": int(hits > 0), "rate": hits / n_words if n_words else 0.0, "n_words": n_words,
            "hits_per_generated_token": hits / n_tokens if n_tokens else 0.0}
