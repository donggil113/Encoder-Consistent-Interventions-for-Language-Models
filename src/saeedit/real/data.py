"""Document-level splits and windows (standard library only).

Order of operations is fixed: segment articles -> assign each article to
calibration or test -> only then tokenize and cut windows. The test document is
the independent unit; windows and positions inside a document are not.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence

TITLE_RE = re.compile(r"^ = [^=].* = $")


@dataclass
class Document:
    doc_id: str
    title: str
    split: str
    text: str
    windows: List[List[int]] = field(default_factory=list)


def segment_wikitext(lines: Iterable[str]) -> List[tuple]:
    """Split raw WikiText lines into ``(title, text)`` articles at level-1 headings."""
    docs: List[tuple] = []
    title, buf = None, []
    for line in lines:
        stripped = line.rstrip("\n")
        if TITLE_RE.match(stripped):
            if title is not None:
                docs.append((title, "".join(buf)))
            title, buf = stripped.strip(" =").strip(), []
        elif title is not None:
            buf.append(line if line.endswith("\n") else line + "\n")
    if title is not None:
        docs.append((title, "".join(buf)))
    return docs


def doc_id(split: str, title: str) -> str:
    return split + ":" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]


def build_documents(split_lines: Dict[str, Iterable[str]]) -> List[Document]:
    """``split_lines`` maps a split name ('calibration'/'test') to raw lines."""
    out = []
    for split, lines in split_lines.items():
        for title, text in segment_wikitext(lines):
            out.append(Document(doc_id(split, title), title, split, text))
    return out


def check_disjoint(docs: Sequence[Document]) -> List[str]:
    """Titles or window hashes present in both splits (must be empty)."""
    titles: Dict[str, set] = {}
    wins: Dict[str, set] = {}
    for d in docs:
        titles.setdefault(d.title, set()).add(d.split)
        for w in d.windows:
            wins.setdefault(hashlib.sha256(repr(w).encode()).hexdigest(), set()).add(d.split)
    bad = [f"title:{t}" for t, s in titles.items() if len(s) > 1]
    bad += [f"window:{h[:12]}" for h, s in wins.items() if len(s) > 1]
    return bad


def make_windows(token_ids: Sequence[int], bos_id: int, window: int = 128, min_len: int = 64) -> List[List[int]]:
    """Consecutive chunks of ``window - 1`` tokens with BOS prepended; short tail dropped."""
    body = window - 1
    out = []
    for start in range(0, len(token_ids), body):
        chunk = list(token_ids[start:start + body])
        if len(chunk) < min_len:
            break
        out.append([bos_id] + chunk)
    return out


def tokenize_documents(docs: Sequence[Document], tokenize: Callable[[str], List[int]], bos_id: int,
                       window: int = 128, min_len: int = 64) -> None:
    for d in docs:
        d.windows = make_windows(tokenize(d.text), bos_id, window, min_len)


def seeded_rng(*parts) -> random.Random:
    return random.Random(":".join(str(p) for p in parts))


def sample_position(window_len: int, rng: random.Random, candidates: Sequence[int] = ()) -> int:
    """A non-BOS position; restricted to ``candidates`` when given."""
    pool = [p for p in candidates if p > 0] if candidates else list(range(1, window_len))
    if not pool:
        raise ValueError("no admissible position")
    return rng.choice(pool)
