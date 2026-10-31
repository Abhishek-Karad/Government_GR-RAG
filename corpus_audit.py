"""Read-only audit of the locally indexed Maharashtra GR corpus."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


DEVANAGARI = re.compile(r"[\u0900-\u097f]")
LATIN = re.compile(r"[A-Za-z]")
NUMBERED = re.compile(r"(?m)^\s*(?:\d+[.)]|[a-z][.)]|\([ivx]+\))\s+")


def normalized(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold(), flags=re.UNICODE).strip()


def audit(db_path: str, config_id: str | None = None) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    docs = con.execute("select * from documents order by source_path").fetchall()
    if config_id is None:
        config_id = con.execute("select config_id from index_configs order by created_at desc limit 1").fetchone()
        config_id = config_id[0] if config_id else None
    texts = [row["gr_text"] or "" for row in docs]
    duplicate_groups = []
    by_text = {}
    for row, text in zip(docs, texts):
        key = normalized(text)
        if key:
            by_text.setdefault(key, []).append(row["document_id"])
    duplicate_groups = [ids for ids in by_text.values() if len(ids) > 1]
    near_duplicates = []
    normalized_docs = [(row["document_id"], normalized(text)) for row, text in zip(docs, texts)]
    for index, (left_id, left) in enumerate(normalized_docs):
        if len(left) < 200:
            continue
        for right_id, right in normalized_docs[index + 1:]:
            if len(right) >= 200:
                ratio = SequenceMatcher(None, left, right).quick_ratio()
                if ratio >= 0.95 and left != right:
                    near_duplicates.append({"document_ids": [left_id, right_id], "ratio": round(ratio, 4)})
    status = {}
    if config_id:
        status = dict(con.execute("select status,count(*) from document_index_status where config_id=? group by status", (config_id,)).fetchall())
    indexed_docs = con.execute("select count(distinct document_id) from document_index_status where config_id=? and status='completed'", (config_id,)).fetchone()[0] if config_id else 0
    audit = {
        "schema_version": "phase0-corpus-audit-v1",
        "database": db_path,
        "selected_index_config_id": config_id,
        "document_count": len(docs),
        "department_count_indexed": len({row["department"] for row in docs if row["department"]}),
        "departments": Counter(row["department"] or "UNKNOWN" for row in docs),
        "source_text": {
            "documents_with_text": sum(bool(text.strip()) for text in texts),
            "total_characters": sum(map(len, texts)),
            "latin_character_count": sum(len(LATIN.findall(text)) for text in texts),
            "devanagari_character_count": sum(len(DEVANAGARI.findall(text)) for text in texts),
            "mixed_script_documents": sum(bool(LATIN.search(text) and DEVANAGARI.search(text)) for text in texts),
        },
        "structure_signals": {
            "documents_with_pipe_table_markers": sum("|" in text for text in texts),
            "documents_with_numbered_or_lettered_clauses": sum(bool(NUMBERED.search(text)) for text in texts),
            "documents_with_ocr_or_table_artifacts": sum(bool(json.loads(row["text_artifacts_json"] or "[]")) for row in docs),
            "artifact_counts": Counter(artifact for row in docs for artifact in json.loads(row["text_artifacts_json"] or "[]")),
        },
        "metadata_completeness": {
            "missing_gr_number": sum(not (row["circular_number"] or "").strip() for row in docs),
            "missing_issue_date": sum(not (row["issue_date"] or "").strip() for row in docs),
            "missing_source_url": sum(not (row["github_url"] or row["official_pdf_url"] or "").strip() for row in docs),
            "gemini_enriched": sum("Gemini" in (row["extraction_source"] or "") for row in docs),
        },
        "duplicates": {
            "exact_duplicate_groups": duplicate_groups,
            "near_duplicate_pairs": near_duplicates,
        },
        "source_availability": {
            "source_paths_recorded": sum(bool(row["source_path"]) for row in docs),
            "distinct_source_paths": len({row["source_path"] for row in docs}),
            "remote_fetch_checked": False,
            "note": "Remote source availability was not probed by this read-only audit.",
        },
        "index_completeness": {
            "documents_with_completed_status": indexed_docs,
            "status_counts": status,
            "chunk_count": con.execute("select count(*) from chunks where index_config_id=?", (config_id,)).fetchone()[0] if config_id else 0,
        },
    }
    for key in ("departments",):
        audit[key] = dict(audit[key])
    for key in ("artifact_counts",):
        audit["structure_signals"][key] = dict(audit["structure_signals"][key])
    con.close()
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="gr_demo.sqlite3")
    parser.add_argument("--config-id")
    parser.add_argument("--json-out", type=Path, default=Path("evaluation/corpus_audit.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path("evaluation/corpus_audit.md"))
    args = parser.parse_args()
    report = audit(args.db, args.config_id)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report["source_text"]; m = report["metadata_completeness"]; i = report["index_completeness"]
    md = ["# Phase 0 corpus audit", "", f"- Documents: {report['document_count']}", f"- Indexed departments in SQLite: {report['department_count_indexed']}", f"- Indexed configuration: `{report['selected_index_config_id']}`", "", "## Indexed departments", ""]
    md.extend(f"- {name}: {count}" for name, count in sorted(report["departments"].items()))
    md += ["", "## Text and structure", "", f"- Characters: {s['total_characters']}", f"- Latin characters: {s['latin_character_count']}", f"- Devanagari characters: {s['devanagari_character_count']}", f"- Mixed-script documents: {s['mixed_script_documents']}", f"- Pipe/table-marker documents: {report['structure_signals']['documents_with_pipe_table_markers']}", f"- Numbered/lettered clause documents: {report['structure_signals']['documents_with_numbered_or_lettered_clauses']}", f"- OCR/table-artifact documents: {report['structure_signals']['documents_with_ocr_or_table_artifacts']}", "", "## Completeness", "", f"- Missing GR number: {m['missing_gr_number']}", f"- Missing issue date: {m['missing_issue_date']}", f"- Missing source URL: {m['missing_source_url']}", f"- Gemini-enriched: {m['gemini_enriched']}", f"- Completed documents in selected index: {i['documents_with_completed_status']}", f"- Chunks in selected index: {i['chunk_count']}", "", "## Limitations", "", "- Remote source availability was not probed; network access was unavailable during this run.", "- This audit covers the local indexed corpus, not an asserted inventory of all 33 remote department folders.", ""]
    args.markdown_out.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
