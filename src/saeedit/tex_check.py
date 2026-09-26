"""Static checks of paper/main.tex when no LaTeX compiler is available.

    PYTHONPATH=src python3 -m saeedit.tex_check --paper paper --out results/reaggregated/tex_check.json

This is NOT a compilation. It checks: citation keys exist in the .bib; every
\\ref/\\cref/\\Cref target is labelled; every macro from generated_numbers.tex that
the text uses is defined; \\input files exist; \\begin/\\end are balanced; no
identifying strings appear; it counts \\todo markers and words in the main body
(an input to a page estimate, not a page count).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

IDENTIFYING = ("github.com/donggil", "donggil", "pusan.ac.kr", "claude.ai", "Claude", "Anthropic")
BUILTIN_OK = set()


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--out", default="results/reaggregated/tex_check.json")
    args = ap.parse_args(argv)
    main_path = os.path.join(args.paper, "main.tex")
    raw = open(main_path).read()
    tex = strip_comments(raw)
    report = {"compiled": False, "note": "static checks only; COMPILE_NOT_RUN (no LaTeX compiler in the environment)"}

    # inputs
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    missing_inputs = []
    body = tex
    for inp in inputs:
        p = os.path.join(args.paper, inp if inp.endswith(".tex") else inp + ".tex")
        if not os.path.exists(p):
            missing_inputs.append(inp)
        else:
            body += "\n" + strip_comments(open(p).read())
    report["inputs"] = inputs
    report["missing_inputs"] = missing_inputs

    # citations
    bib = open(os.path.join(args.paper, "references.bib")).read()
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    cites = set()
    for grp in re.findall(r"\\(?:citep|citet|cite|citealp|yrcite|citeauthor)\*?(?:\[[^\]]*\])*\{([^}]+)\}", tex):
        cites.update(k.strip() for k in grp.split(","))
    report["citations_used"] = sorted(cites)
    report["citations_missing_in_bib"] = sorted(cites - bib_keys)
    report["bib_entries_unused"] = sorted(bib_keys - cites)

    # labels / refs
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set()
    for grp in re.findall(r"\\(?:ref|cref|Cref|eqref|autoref)\{([^}]+)\}", tex):
        refs.update(k.strip() for k in grp.split(","))
    legend_names = set(re.findall(r"legend to name=([A-Za-z0-9_:-]+)", tex))
    report["refs_undefined"] = sorted(refs - labels - legend_names)
    report["labels_unreferenced"] = sorted(labels - refs)

    # generated number macros
    gen = open(os.path.join(args.paper, "generated_numbers.tex")).read()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", gen))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)\b", tex))
    own = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex))
    known_latex = {"Cref", "R", "LN", "LaTeX"}
    cand = {u for u in used if (u.startswith(("R", "D", "S", "T", "V", "E", "F")) and u not in known_latex and u not in own)}
    report["number_macros_used"] = sorted(cand & defined)
    report["number_macros_undefined"] = sorted(u for u in cand - defined
                                               if re.match(r"^(R(one|two|three|foura|fourb|five|zero)|Dense|Sparse|Tx|Tr|Vfive|Vsix|Enc|Feas|Repair|Topk)", u))

    # environments
    begins = Counter(re.findall(r"\\begin\{([^}]+)\}", tex))
    ends = Counter(re.findall(r"\\end\{([^}]+)\}", tex))
    report["unbalanced_environments"] = {k: [begins[k], ends[k]] for k in set(begins) | set(ends) if begins[k] != ends[k]}

    # anonymity
    report["identifying_strings_found"] = [s for s in IDENTIFYING if s in raw]
    report["accepted_option_used"] = bool(re.search(r"\\usepackage\[accepted\]\{icml2026\}", tex))

    # todos and length
    report["todo_markers"] = re.findall(r"\\todo\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", tex)
    m1 = tex.find(r"\begin{abstract}")
    m2 = tex.find(r"\section*{Impact Statement}")
    main_body = tex[m1:m2] if m1 >= 0 and m2 > m1 else ""
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", re.sub(r"\\[A-Za-z]+\*?", " ", main_body))
    report["main_body_words_approx"] = len(words)
    report["page_estimate_note"] = "word count includes table/figure code tokens loosely; true page count needs compilation"
    report["ok"] = not (missing_inputs or report["citations_missing_in_bib"] or report["refs_undefined"]
                        or report["number_macros_undefined"] or report["unbalanced_environments"]
                        or report["identifying_strings_found"] or report["accepted_option_used"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("ok", "missing_inputs", "citations_missing_in_bib", "refs_undefined",
                                             "number_macros_undefined", "unbalanced_environments",
                                             "identifying_strings_found", "main_body_words_approx")}, indent=1))
    print("todo markers:", len(report["todo_markers"]))
    print("unused bib:", report["bib_entries_unused"])
    print("unreferenced labels:", report["labels_unreferenced"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
