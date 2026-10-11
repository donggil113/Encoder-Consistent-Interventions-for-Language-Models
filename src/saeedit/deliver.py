"""R8 delivery runner (standard library; pymupdf is optional and only used for page facts).

    PYTHONPATH=src python3 -m saeedit.deliver --config configs/auto_run_r8.json \
        [--texbin DIR] [--scratch DIR]

The config is the frozen contract (run id, allowed stages, input hashes, limits). The runner
verifies the built PDF against the recorded hashes, copies it byte-for-byte into
deliverables/, writes the source zip with a BUILD.md, optionally checks that the zip builds on
its own in a scratch folder (paper/main.pdf is never rebuilt or replaced) and writes the
delivery record. A stage whose recorded evidence still matches its inputs is skipped. Only
the "evidence" block of the config is written back; the frozen part is left as it is. No
experiment runs here and nothing is sent anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import zipfile

TEXT_EXT = (".tex", ".bib", ".sty", ".bst", ".dat", ".md", ".txt", ".csv", ".json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def frozen_sha(cfg):
    """Hash of the contract without the runner-written evidence block."""
    return sha256_bytes(json.dumps({k: v for k, v in cfg.items() if k != "evidence"}, sort_keys=True).encode())


def git_blob(path, root):
    p = subprocess.run(["git", "hash-object", path], cwd=root, stdout=subprocess.PIPE, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def pdf_facts(path, marker):
    """Page count, body-end page (first page holding the conclusion's last sentence), metadata, texts."""
    try:
        import pymupdf  # noqa
    except ImportError:
        return {"status": "NOT_RUN", "reason": "pymupdf not importable"}
    doc = pymupdf.open(path)
    texts, body_end, appendix = [], None, None
    for i, page in enumerate(doc):
        t = page.get_text()
        texts.append(" ".join(t.split()))
        if body_end is None and marker in t:
            body_end = i + 1
        if appendix is None and "A. Standard Facts" in t:
            appendix = i + 1
    return {"status": "OK", "pages": doc.page_count, "body_end_page": body_end, "appendix_start_page": appendix,
            "page_size_pt": [round(doc[0].rect.width, 1), round(doc[0].rect.height, 1)],
            "metadata": {k: doc.metadata.get(k) for k in ("title", "author", "producer", "creator")},
            "texts": texts}


def scan_text(data, patterns, commits):
    txt = data.decode("utf-8", errors="replace")
    hits = []
    if patterns:
        rx = re.compile("|".join(re.escape(p) for p in patterns), re.IGNORECASE)
        hits += [{"line": txt.count("\n", 0, m.start()) + 1, "match": m.group(), "kind": "identifying"} for m in rx.finditer(txt)]
    if commits:
        crx = re.compile(r"\b(" + "|".join(re.escape(c) for c in commits) + r")\b")
        hits += [{"line": txt.count("\n", 0, m.start()) + 1, "match": m.group(), "kind": "commit id"} for m in crx.finditer(txt)]
    return hits


def read_list(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        s = line.strip()
        if s and not s.startswith("#"):
            out += s.split()
    return out


def build_md(cfg, pdf):
    c = cfg
    lines = [f"# {c['project']} manuscript source ({c['manuscript_version']}, source commit {c['source']['head'][:7]})", "",
             f"Delivered PDF: `{c['deliverables']['pdf_name']}` (byte-identical to `paper/main.pdf` at commit "
             f"`{c['source']['head']}` on branch `{c['source']['branch']}`): sha256 `{pdf['sha256']}`, {pdf['bytes']:,} bytes, "
             f"{pdf['pages']} pages (US letter), body (last sentence of the Conclusion) ends on page {pdf['body_end_page']}; "
             f"the one-column appendix starts on page {pdf['appendix_start_page']}.", "",
             "## Style", "", c["style"]["description"], "",
             "## Build that produced the delivered PDF (recorded)", "", c["build"]["recorded"]["description"], "", "```"]
    lines += c["build"]["recorded"]["commands"] + ["```", "", "## Reproduction command (used for the R8 self-containment check of this zip)", "", "```"]
    lines += c["build"]["check"]["shell"] + ["```", "",
             "A byte-identical PDF is not expected from a rebuild (pdfTeX embeds the creation date and a document id); "
             "the comparison is the page count and the extracted text of every page.", "",
             "## Contents", "", "`paper/`: " + ", ".join(os.path.relpath(m, "paper") for m in c["source_zip"]["members"]) + ".", "",
             c["source_zip"]["not_included"], "",
             "## Required LaTeX packages", "", ", ".join(c["build"]["required_packages"]) + ".", "",
             "## Status", ""]
    rs = c["research_status"]
    lines += [f"- Study: {rs['study']}; {rs['method_utility']}.",
              f"- SUBMISSION_READY = {str(rs['SUBMISSION_READY']).lower()}; TARGET_YEAR = {rs['TARGET_YEAR']}, TEMPLATE_YEAR = {rs['TEMPLATE_YEAR']} ({rs['icml_2027_instructions']}).",
              f"- External read review: {rs['external_read_review']}.",
              "- Compile status, visual check, scientific validity, novelty and submission readiness are separate records "
              "(see P3_delivery.json); this file certifies none of them beyond the build facts above.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/auto_run_r8.json")
    ap.add_argument("--root", default=".")
    ap.add_argument("--texbin", default=None, help="directory with pdflatex/bibtex for the optional self-containment check")
    ap.add_argument("--scratch", default=None, help="scratch directory for the optional self-containment check")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    cfg_path = os.path.join(root, args.config)
    cfg = json.load(open(cfg_path))
    frozen = frozen_sha(cfg)
    ev = cfg.setdefault("evidence", {})
    t0, r0 = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    out_dir = os.path.join(root, cfg["deliverables"]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    report = {"run_id": cfg["run_id"], "stages": {}}

    # S1 verify the built PDF against the frozen record
    src_pdf = os.path.join(root, cfg["inputs_primary"]["pdf"])
    exp = cfg["inputs"][cfg["inputs_primary"]["pdf"]]
    head = open(src_pdf, "rb").read(8)
    facts = pdf_facts(src_pdf, cfg["pdf_markers"]["conclusion_last_sentence"])
    s1 = {"pdf_header": head.decode("latin-1"), "bytes": os.path.getsize(src_pdf), "sha256": sha256_file(src_pdf),
          "git_blob_sha1": git_blob(cfg["inputs_primary"]["pdf"], root),
          "expected": {k: exp.get(k) for k in ("sha256", "bytes", "git_blob_sha1", "pages", "body_end_page")}}
    s1.update({k: facts.get(k) for k in ("pages", "body_end_page", "appendix_start_page", "page_size_pt", "metadata", "status")})
    ok = (s1["sha256"] == exp["sha256"] and s1["bytes"] == exp["bytes"] and head.startswith(b"%PDF-")
          and (facts.get("status") != "OK" or (facts["pages"] == exp["pages"] and facts["body_end_page"] == exp["body_end_page"])))
    s1["verdict"] = "MATCH" if ok else "STOP_MISMATCH"
    report["stages"]["verify_pdf"] = s1
    if not ok:
        print(json.dumps(report, indent=1))
        return 1
    for rel, h in cfg["inputs"].items():
        if "sha256" in h and sha256_file(os.path.join(root, rel)) != h["sha256"]:
            report["stages"]["verify_pdf"] = {"verdict": "STOP_MISMATCH", "path": rel}
            print(json.dumps(report, indent=1))
            return 1
    ev["verify_pdf"] = {"sha256": s1["sha256"], "verdict": "MATCH"}

    # S2 byte copy of the PDF
    pdf_name = cfg["deliverables"]["pdf_name"]
    dst = os.path.join(out_dir, pdf_name)
    if os.path.exists(dst) and sha256_file(dst) == s1["sha256"]:
        s2 = {"status": "SKIPPED (present, hash matches)"}
    else:
        shutil.copyfile(src_pdf, dst)
        os.chmod(dst, 0o644)
        s2 = {"status": "COPIED"}
    s2.update({"file": os.path.relpath(dst, root), "sha256": sha256_file(dst), "bytes": os.path.getsize(dst),
               "byte_identical_to_source": sha256_file(dst) == s1["sha256"] and os.path.getsize(dst) == s1["bytes"]})
    report["stages"]["copy_pdf"] = s2
    ev["copy_pdf"] = {"file": s2["file"], "sha256": s2["sha256"]}

    # S3 source zip (deterministic: sorted members, fixed timestamps, no extra attributes)
    members = cfg["source_zip"]["members"]
    pdf_for_md = {"sha256": s1["sha256"], "bytes": s1["bytes"], "pages": facts.get("pages", exp["pages"]),
                  "body_end_page": facts.get("body_end_page", exp["body_end_page"]),
                  "appendix_start_page": facts.get("appendix_start_page", exp.get("appendix_start_page"))}
    md = build_md(cfg, pdf_for_md).encode("utf-8")
    md_path = os.path.join(out_dir, cfg["deliverables"]["build_md_name"])
    with open(md_path, "wb") as f:
        f.write(md)
    patterns = read_list(os.path.join(root, cfg["source_zip"]["scan"]["patterns_file"]))
    commits = sorted({c for c in read_list(os.path.join(root, cfg["source_zip"]["scan"]["commits_file"]))}, key=len, reverse=True)
    entries, hits, buf = [], [], io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("BUILD.md", date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o644 << 16
        z.writestr(info, md)
        entries.append({"path": "BUILD.md", "bytes": len(md), "sha256": sha256_bytes(md)})
        for rel in sorted(members):
            data = open(os.path.join(root, rel), "rb").read()
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
            entries.append({"path": rel, "bytes": len(data), "sha256": sha256_bytes(data)})
            if rel.endswith(TEXT_EXT):
                for h in scan_text(data, patterns, commits):
                    hits.append(dict(path=rel, **h))
    zip_bytes = buf.getvalue()
    zip_path = os.path.join(out_dir, cfg["deliverables"]["zip_name"])
    zip_sha = sha256_bytes(zip_bytes)
    if os.path.exists(zip_path) and sha256_file(zip_path) == zip_sha:
        zstatus = "SKIPPED (present, identical bytes)"
    else:
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        zstatus = "WRITTEN"
    s3 = {"status": zstatus, "file": os.path.relpath(zip_path, root), "sha256": zip_sha, "bytes": len(zip_bytes),
          "n_members": len(entries), "members": entries,
          "identifying_string_scan": {"patterns_from": cfg["source_zip"]["scan"]["patterns_file"],
                                      "n_patterns": len(patterns), "n_commit_ids": len(commits), "hits": hits,
                                      "note": "string check of the shipped text files; the zip is a user deliverable and is not redacted"}}
    report["stages"]["source_zip"] = s3
    ev["source_zip"] = {"file": s3["file"], "sha256": zip_sha}

    # S4 optional self-containment build of the zip in a scratch folder (never touches paper/)
    bc = cfg["build"]  # check commands live under build.commands in the contract
    prev = ev.get("build_check", {})
    if prev.get("zip_sha256") == zip_sha and prev.get("verdict") == "PASS":
        s4 = {"status": "SKIPPED (same zip already checked)", **prev}
    elif not args.texbin or not os.path.exists(os.path.join(args.texbin, "pdflatex")) or facts.get("status") != "OK":
        s4 = {"status": "NOT_RUN", "reason": "no --texbin with pdflatex given, or pymupdf unavailable"}
    else:
        work = os.path.join(args.scratch or os.path.join(root, "deliverables", "_check"), "r8_zipcheck")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(work)
        pdir = os.path.join(work, "paper")
        env = dict(os.environ, PATH=args.texbin + os.pathsep + os.environ.get("PATH", ""))
        rc0, w0 = resource.getrusage(resource.RUSAGE_CHILDREN), time.monotonic()
        runs = []
        for argv in bc["commands"]:
            p = subprocess.run(argv, cwd=pdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=cfg["limits"]["cpu_s_structural_check"])
            runs.append({"argv": argv, "returncode": p.returncode, "tail": p.stdout.decode(errors="replace")[-300:]})
            if p.returncode != 0:
                break
        rc1, w1 = resource.getrusage(resource.RUSAGE_CHILDREN), time.monotonic()
        log_path = os.path.join(pdir, "main.log")
        log = open(log_path, errors="replace").read() if os.path.exists(log_path) else ""
        built = os.path.join(pdir, "main.pdf")
        bf = pdf_facts(built, cfg["pdf_markers"]["conclusion_last_sentence"]) if os.path.exists(built) else {"status": "NOT_BUILT"}
        diff_pages = ([i + 1 for i, (a, b) in enumerate(zip(facts["texts"], bf.get("texts", []))) if a != b]
                      if bf.get("status") == "OK" else None)
        s4 = {"status": "RAN", "work_dir": work, "commands": runs,
              "wall_s": round(w1 - w0, 2),
              "cpu_s_children": round((rc1.ru_utime - rc0.ru_utime) + (rc1.ru_stime - rc0.ru_stime), 2),
              "log": {"write18": re.search(r"\\write18 (enabled|disabled)\.|restricted \\write18 enabled\.", log).group(0) if re.search(r"write18", log) else None,
                      "overfull_boxes": len(re.findall(r"^Overfull \\[hv]box", log, re.M)),
                      "undefined_refs_or_cites": len(re.findall(r"LaTeX Warning: (Reference|Citation) .* undefined|There were undefined", log)),
                      "font_shape_warnings": len(re.findall(r"LaTeX Font Warning", log))},
              "built_pdf": {k: bf.get(k) for k in ("status", "pages", "body_end_page", "appendix_start_page", "metadata")},
              "text_differs_on_pages": diff_pages,
              "verdict": ("PASS" if all(r["returncode"] == 0 for r in runs) and bf.get("status") == "OK"
                          and bf["pages"] == facts["pages"] and diff_pages == [] else "FAIL"),
              "zip_sha256": zip_sha,
              "note": "check only; paper/main.pdf and the delivered PDF are not replaced"}
        ev["build_check"] = {k: s4[k] for k in ("verdict", "zip_sha256", "wall_s", "cpu_s_children")}
    report["stages"]["build_check"] = s4

    # S5 delivery record
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    files = []
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p) and name != cfg["deliverables"]["record_name"]:
            files.append({"path": os.path.relpath(p, root), "bytes": os.path.getsize(p), "sha256": sha256_file(p)})
    delivery = {
        "project": cfg["project"], "manuscript_version": cfg["manuscript_version"], "run_id": cfg["run_id"],
        "date": cfg["date"], "source_commit": cfg["source"],
        "pdf": {"file": s2["file"], "copied_from": cfg["inputs_primary"]["pdf"], "byte_identical_to_source": s2["byte_identical_to_source"],
                "sha256": s2["sha256"], "bytes": s2["bytes"], "git_blob_sha1": s1["git_blob_sha1"], "pdf_header": s1["pdf_header"],
                "pages": s1["pages"], "body_end_page": s1["body_end_page"], "appendix_start_page": s1["appendix_start_page"],
                "page_size_pt": s1["page_size_pt"], "metadata": s1["metadata"], "page_facts_source": s1["status"]},
        "immutable_links": cfg["immutable_links"],
        "style": cfg["style"],
        "build": {"recorded": cfg["build"]["recorded"], "self_containment_check": {k: v for k, v in s4.items() if k != "commands"},
                  "check_commands": s4.get("commands")},
        "pages_viewed": cfg["pages_viewed"],
        "research_status": cfg["research_status"],
        "source_zip": s3,
        "files": files,
        "cost": {"wall_s": round(time.monotonic() - t0, 2),
                 "cpu_s_self": round((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), 2),
                 "cpu_s_children": s4.get("cpu_s_children", 0.0), "network": "none inside the runner", "gpu": "none"},
        "runner": {"module": "saeedit.deliver", "sha256": sha256_file(os.path.abspath(__file__)),
                   "config": args.config, "config_frozen_sha256": frozen},
    }
    rec_path = os.path.join(out_dir, cfg["deliverables"]["record_name"])
    with open(rec_path, "w") as f:
        json.dump(delivery, f, indent=1)
    ev["delivery_record"] = {"file": os.path.relpath(rec_path, root), "sha256": sha256_file(rec_path)}
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    report["stages"]["delivery_record"] = ev["delivery_record"]
    report["cost"] = delivery["cost"]
    brief = {"verify_pdf": s1["verdict"], "copy_pdf": s2["status"], "source_zip": zstatus,
             "build_check": s4.get("verdict", s4.get("status")), "record": ev["delivery_record"], "cost": delivery["cost"],
             "scan_hits": len(hits)}
    print(json.dumps(brief, indent=1))
    return 0 if s4.get("verdict", "PASS") == "PASS" and not hits else 2


if __name__ == "__main__":
    sys.exit(main())
