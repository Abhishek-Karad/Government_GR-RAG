#!/usr/bin/env python3
"""Generate and verify a balanced 100-case evaluation set from SQLite.

The generated questions are intentionally deterministic. Every evidence marker
and fact is copied from current indexed source chunks; this script does not use
an LLM to invent annotations.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "gr_demo.sqlite3"
SEED_PATH = ROOT / "evaluation" / "seed_cases.jsonl"
OUT_PATH = ROOT / "evaluation" / "verified_cases_100.jsonl"
REPORT_PATH = ROOT / "evaluation" / "verified_cases_report.json"

TARGETS = {
    "Law and Judiciary Department": 45,
    "Agriculture, Dairy Development, Animal Husbandry and Fisheries Department": 30,
    "Rural Development Department": 20,
    "Medical Education and Drugs Department": 3,
}


def norm(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().casefold()


def clean_marker(text: str, limit: int = 150) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -|\t")
    if len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        text = text[:cut if cut > 40 else limit].rstrip(" ,;:")
    return text


def number_token(text: str) -> str | None:
    match = re.search(r"(?<![A-Za-z])(?:Rs\.?\s*)?\d[\d, ]*(?:\.\d+)?", text)
    if not match:
        return None
    value = match.group(0).strip()
    value = re.sub(r"^Rs\.?\s*", "", value, flags=re.I)
    return value.strip()


def load_existing() -> list[dict]:
    cases = [json.loads(line) for line in SEED_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    for case in cases:
        # Correct the known duplicate annotation without changing its meaning.
        case["expected_document_ids"] = list(dict.fromkeys(case.get("expected_document_ids", [])))
        if "expected_pages" not in case:
            pages = sorted({evidence.get("page") for evidence in case.get("expected_evidence", [])
                            if evidence.get("page") is not None})
            if pages:
                case["expected_pages"] = pages
    return cases


def current_config(conn: sqlite3.Connection) -> str:
    row = conn.execute("""SELECT config_id FROM index_configs
                         WHERE chunking_strategy='structure-aware-v3-contextual-hybrid'
                         ORDER BY created_at DESC LIMIT 1""").fetchone()
    if not row:
        raise RuntimeError("Current contextual-hybrid index configuration was not found")
    return row[0]


def load_documents(conn: sqlite3.Connection, config_id: str) -> dict[str, dict]:
    rows = conn.execute("""
        SELECT d.document_id,d.department,d.filename,d.circular_number,d.issue_date,
               d.sha,d.index_version,s.source_sha,s.index_version
        FROM documents d
        JOIN document_index_status s ON s.document_id=d.document_id AND s.config_id=?
        WHERE d.index_status='completed' AND s.status='completed'
          AND d.sha=s.source_sha AND (d.index_version IS NULL OR s.index_version=d.index_version)
        ORDER BY d.department,d.document_id
    """, (config_id,)).fetchall()
    return {
        row[0]: {
            "document_id": row[0], "department": row[1], "filename": row[2],
            "circular": row[3], "issue_date": row[4], "sha": row[5],
            "index_version": row[6],
        }
        for row in rows
    }


def load_chunks(conn: sqlite3.Connection, config_id: str) -> dict[str, list[dict]]:
    rows = conn.execute("""
        SELECT c.document_id,c.chunk_index,c.page_start,c.page_end,c.text,
               d.department,d.filename,d.circular_number,d.issue_date
        FROM chunks c JOIN documents d ON d.document_id=c.document_id
        JOIN document_index_status s
          ON s.document_id=c.document_id AND s.config_id=c.index_config_id
        WHERE c.index_config_id=? AND d.index_status='completed' AND s.status='completed'
          AND d.sha=s.source_sha AND (d.index_version IS NULL OR s.index_version=d.index_version)
        ORDER BY c.document_id,c.chunk_index
    """, (config_id,)).fetchall()
    result: dict[str, list[dict]] = {}
    for row in rows:
        result.setdefault(row[0], []).append({
            "document_id": row[0], "chunk_index": row[1], "page_start": row[2],
            "page_end": row[3], "text": row[4], "department": row[5],
            "filename": row[6], "circular": row[7], "issue_date": row[8],
        })
    return result


def valid_marker(chunks: list[dict], marker: str) -> bool:
    target = norm(marker)
    return bool(target) and any(target in norm(chunk["text"]) for chunk in chunks)


def page_label(chunk: dict) -> int | str:
    return chunk["page_start"] if chunk["page_start"] == chunk["page_end"] else f"{chunk['page_start']}-{chunk['page_end']}"


def choose_marker(chunk: dict) -> str | None:
    lines = [clean_marker(line, 150) for line in chunk["text"].splitlines() if clean_marker(line, 150)]
    for line in lines:
        if len(line) >= 35 and not re.fullmatch(r"[-_=| .:]+", line):
            return line
    text = clean_marker(chunk["text"], 150)
    return text if len(text) >= 20 else None


def make_case(case_id: str, doc: dict, chunks: list[dict], ordinal: int) -> dict | None:
    usable = []
    for chunk in chunks:
        marker = choose_marker(chunk)
        if marker and valid_marker(chunks, marker):
            usable.append((chunk, marker))
    if not usable:
        return None
    chunk, marker = usable[ordinal % len(usable)]
    circular = doc["circular"]
    reference = circular or Path(doc["filename"]).stem
    mode = ordinal % 5
    if mode == 0 and circular:
        question = f"What does government resolution {reference} state about the following provision: {marker}?"
        question_type = "exact_identifier"
    elif mode == 1:
        question = f"What requirement or decision is recorded in the {doc['department']} document about {marker.lower()}?"
        question_type = "policy_requirement"
    elif mode == 2:
        question = f"What procedure or administrative action is described in the {doc['department']} document concerning {marker.lower()}?"
        question_type = "procedure"
    elif mode == 3 and number_token(marker):
        question = f"What numerical value is stated in the {doc['department']} document in connection with {marker.lower()}?"
        question_type = "numerical"
    else:
        question = f"According to the {doc['department']} document, what is stated about {marker.lower()}?"
        question_type = "multi_aspect"
    fact_value = number_token(marker) if question_type == "numerical" else marker
    fact_type = "number" if question_type == "numerical" else "text"
    evidence_id = f"source_{case_id}"
    return {
        "id": case_id,
        "question": question,
        "departments": [doc["department"]],
        "expected_document_ids": [doc["document_id"]],
        "expected_gr_numbers": [circular] if circular else [],
        "expected_pages": [chunk["page_start"]],
        "expected_evidence": [{
            "id": evidence_id, "document_id": doc["document_id"],
            "page": chunk["page_start"], "text_contains": marker,
        }],
        "expected_facts": [{"value": fact_value, "type": fact_type, "evidence_ids": [evidence_id]}],
        "answerable": True,
        "question_type": question_type,
        "verification_note": "Generated from and verified against the current contextual-hybrid SQLite source chunk.",
    }


def make_cross_case(case_id: str, left: tuple[dict, dict], right: tuple[dict, dict]) -> dict:
    left_doc, left_chunk = left
    right_doc, right_chunk = right
    left_marker = choose_marker(left_chunk) or clean_marker(left_chunk["text"])
    right_marker = choose_marker(right_chunk) or clean_marker(right_chunk["text"])
    left_id, right_id = f"source_{case_id}_left", f"source_{case_id}_right"
    return {
        "id": case_id,
        "question": f"Compare what the {left_doc['department']} document states about {left_marker.lower()} with what the {right_doc['department']} document states about {right_marker.lower()}.",
        "departments": [left_doc["department"], right_doc["department"]],
        "expected_document_ids": [left_doc["document_id"], right_doc["document_id"]],
        "expected_gr_numbers": [x for x in (left_doc["circular"], right_doc["circular"]) if x],
        "expected_pages": [left_chunk["page_start"], right_chunk["page_start"]],
        "expected_evidence": [
            {"id": left_id, "document_id": left_doc["document_id"], "page": left_chunk["page_start"], "text_contains": left_marker},
            {"id": right_id, "document_id": right_doc["document_id"], "page": right_chunk["page_start"], "text_contains": right_marker},
        ],
        "expected_facts": [
            {"value": left_marker, "type": "text", "evidence_ids": [left_id]},
            {"value": right_marker, "type": "text", "evidence_ids": [right_id]},
        ],
        "answerable": True,
        "question_type": "cross_document",
        "verification_note": "Both comparison markers were verified against current contextual-hybrid SQLite chunks.",
    }


def verify_case(case: dict, chunks_by_doc: dict[str, list[dict]]) -> list[str]:
    errors = []
    for evidence in case.get("expected_evidence", []):
        chunks = chunks_by_doc.get(evidence.get("document_id"), [])
        if not valid_marker(chunks, evidence.get("text_contains", "")):
            errors.append(f"{evidence.get('id')}: marker not found in current chunks")
        else:
            matching = [chunk for chunk in chunks if norm(evidence["text_contains"]) in norm(chunk["text"])]
            if not any(str(chunk["page_start"]) == str(evidence.get("page")) for chunk in matching):
                errors.append(f"{evidence.get('id')}: page does not match source chunk")
    evidence_ids = {x.get("id") for x in case.get("expected_evidence", [])}
    for fact in case.get("expected_facts", []):
        if not set(fact.get("evidence_ids", [])) <= evidence_ids:
            errors.append(f"fact references missing evidence: {fact}")
    return errors


def main() -> int:
    conn = sqlite3.connect(DB_PATH)
    config_id = current_config(conn)
    documents = load_documents(conn, config_id)
    chunks = load_chunks(conn, config_id)
    existing = load_existing()
    existing_ids = {x["id"] for x in existing}
    used_docs = {doc_id for case in existing for doc_id in case.get("expected_document_ids", [])}

    generated: list[dict] = []
    skipped: list[dict] = []
    counters = Counter()
    next_number = 10
    for department, target in TARGETS.items():
        already = sum(1 for case in existing if department in case.get("departments", []))
        needed = target - already
        candidates = [doc for doc in documents.values()
                      if doc["department"] == department
                      and (department == "Medical Education and Drugs Department"
                           or doc["document_id"] not in used_docs)
                      and chunks.get(doc["document_id"])]
        if department == "Medical Education and Drugs Department" and candidates:
            # The corpus has only two medical documents; allow a second
            # supported question on one document when three are requested.
            candidates = (candidates * ((needed + len(candidates) - 1) // len(candidates)))[:needed * 3]
        for ordinal, doc in enumerate(candidates[:max(0, needed * 3)]):
            if counters[department] >= needed:
                break
            case_id = f"eval_{next_number:03d}"
            next_number += 1
            case = make_case(case_id, doc, chunks[doc["document_id"]], ordinal)
            if not case:
                skipped.append({"id": case_id, "reason": "no usable source marker", "document_id": doc["document_id"]})
                continue
            errors = verify_case(case, chunks)
            if errors:
                skipped.append({"id": case_id, "reason": errors, "document_id": doc["document_id"]})
                continue
            generated.append(case)
            counters[department] += 1
            used_docs.add(doc["document_id"])

    # One supported cross-department comparison completes the two cross/unanswerable slots
    # together with the preserved unanswerable regression case.
    law_doc = next(doc for doc in documents.values() if doc["department"] == "Law and Judiciary Department" and doc["document_id"] not in used_docs)
    agri_doc = next(doc for doc in documents.values() if doc["department"].startswith("Agriculture,") and doc["document_id"] not in used_docs)
    left = (law_doc, chunks[law_doc["document_id"]][0])
    right = (agri_doc, chunks[agri_doc["document_id"]][0])
    cross = make_cross_case(f"eval_{next_number:03d}", left, right)
    next_number += 1
    cross_errors = verify_case(cross, chunks)
    if not cross_errors:
        generated.append(cross)
    else:
        skipped.append({"id": cross["id"], "reason": cross_errors})

    all_cases = existing + generated
    errors_by_case = {case["id"]: verify_case(case, chunks) for case in all_cases}
    invalid_existing = {case_id: errors for case_id, errors in errors_by_case.items() if errors}
    if invalid_existing:
        raise SystemExit(json.dumps({"invalid_existing_or_generated": invalid_existing}, indent=2))
    if len(all_cases) != 100:
        raise SystemExit(f"Generated {len(all_cases)} cases instead of 100; counts={dict(counters)}")

    OUT_PATH.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in all_cases) + "\n", encoding="utf-8")
    report = {
        "source_database": str(DB_PATH), "config_id": config_id,
        "case_count": len(all_cases), "generated_count": len(generated),
        "preserved_count": len(existing), "skipped_candidates": skipped,
        "department_counts": dict(Counter(dept for case in all_cases for dept in case.get("departments", []))),
        "target_distribution": {
            "Law and Judiciary Department": 45,
            "Agriculture, Dairy Development, Animal Husbandry and Fisheries Department": 30,
            "Rural Development Department": 20,
            "Medical Education and Drugs Department": 3,
            "cross_department_or_unanswerable": 2,
        },
        "department_specific_counts": {
            department: sum(1 for case in all_cases
                            if len(case.get("departments", [])) == 1
                            and case.get("departments", [None])[0] == department)
            for department in TARGETS
        },
        "cross_department_or_unanswerable_count": sum(
            1 for case in all_cases
            if not case.get("answerable", True) or len(case.get("departments", [])) > 1),
        "unverified_cases": [],
        "current_index_document_count": len(documents),
        "current_index_chunk_count": sum(len(items) for items in chunks.values()),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
