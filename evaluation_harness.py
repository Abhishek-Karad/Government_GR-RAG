"""FastAPI-only Phase 0 evaluation client.

This process is an external API consumer. It never imports the CLI RAG path or
accesses SQLite, Qdrant, embedding, reranking, selection, or Gemini internals.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests


METRIC_DEFINITIONS = {
    "document_recall_at_30": "Fraction of required documents in the original pre-rerank candidate diagnostics, limited to rank 30.",
    "candidate_evidence_recall_at_30": "Fraction of verified evidence specifications matched by the fused candidate diagnostics through rank 30.",
    "document_all_retrieved_at_30": "True only when every required document is in the original candidate set; null for empty gold.",
    "evidence_recall_at_6": "Fraction of verified evidence specifications matched by selected API evidence; document membership alone is insufficient.",
    "precision_at_6": "Fraction of selected API passages matching verified evidence specifications.",
    "mrr": "Reciprocal rank of the first required document in the original pre-rerank candidate order.",
    "fact_coverage": "Fraction of evidence-linked facts whose typed value appears in the matched selected passage.",
}


def normalize(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").casefold().replace("–", "-").replace("—", "-"))


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (value or "").casefold())).strip()


def numeric_values(value: str | None) -> set[Decimal]:
    matches = re.findall(r"(?<![A-Za-z0-9])\d(?:[\d, ]*\d)?(?:\.\d+)?(?![A-Za-z0-9])", value or "")
    result = set()
    for match in matches:
        try:
            result.add(Decimal(re.sub(r"[\s,]", "", match)))
        except InvalidOperation:
            pass
    return result


def canonical_date(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", value.casefold()).strip()
    for pattern in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    return None


def value_matches(value: str, text: str, value_type: str = "text") -> bool:
    if value_type == "number":
        return bool(numeric_values(value) & numeric_values(text))
    if value_type == "date":
        expected = canonical_date(value)
        date_patterns = r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{1,2}(?:st|nd|rd|th)? [A-Za-z]+ \d{4}|[A-Za-z]+ \d{1,2},? \d{4}"
        return bool(expected and any(canonical_date(candidate) == expected for candidate in re.findall(date_patterns, text)))
    return normalize_text(value) in normalize_text(text)


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not case.get("id") or not case.get("question"):
            raise ValueError(f"{path}:{line_no}: id and question are required")
        cases.append(case)
    return cases


class ApiEvaluationError(RuntimeError):
    pass


class FastApiClient:
    def __init__(self, base_url: str, timeout: float, retries: int):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(0, retries)
        self.session = requests.Session()

    def health(self) -> dict:
        response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        return self._json(response)

    def query(self, question: str, *, candidate_k: int, top_k: int,
              max_per_document: int, rerank: bool, generate_answer: bool,
              diagnostics: bool) -> dict:
        payload = {
            "question": question, "candidate_k": candidate_k, "top_k": top_k,
            "max_per_document": max_per_document, "rerank": rerank,
            "show_evidence": True, "generate_answer": generate_answer,
            "evaluation_diagnostics": diagnostics,
        }
        last = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                response = self.session.post(f"{self.base_url}/api/v1/query", json=payload, timeout=self.timeout)
                body = self._json(response)
                body["client_total_ms"] = (time.perf_counter() - started) * 1000
                body["request_parameters"] = payload
                return body
            except (requests.RequestException, ApiEvaluationError) as exc:
                last = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt >= self.retries or (status is not None and status not in {408, 429, 500, 502, 503, 504}):
                    raise
                time.sleep(min(2.0, 0.25 * (2 ** attempt)))
        raise last or ApiEvaluationError("API request failed")

    @staticmethod
    def _json(response) -> dict:
        try:
            body = response.json()
        except ValueError as exc:
            raise ApiEvaluationError(f"invalid JSON response (HTTP {response.status_code})") from exc
        if response.status_code >= 400:
            error = ApiEvaluationError(f"HTTP {response.status_code}: {body}")
            error.response = response
            raise error
        if not isinstance(body, dict):
            raise ApiEvaluationError("API response is not a JSON object")
        return body


def evidence_matches(item: dict, spec: dict) -> bool:
    if spec.get("document_id") and item.get("document_id") != spec["document_id"]:
        return False
    if spec.get("page") is not None:
        page = int(spec["page"])
        start = int(item.get("page_start") or item.get("page_label", "0").split("-")[0])
        end = int(item.get("page_end") or item.get("page_label", "0").split("-")[-1])
        if not start <= page <= end:
            return False
    if spec.get("text_contains") and normalize_text(spec["text_contains"]) not in normalize_text(item.get("text")):
        return False
    return True


def gold_documents(case: dict, candidates: list[dict], selected: list[dict]) -> set[str]:
    gold = set(case.get("expected_document_ids") or [])
    gold.update(spec.get("document_id") for spec in case.get("expected_evidence") or [] if spec.get("document_id"))
    needles = [normalize(x) for x in case.get("expected_gr_numbers") or []]
    for item in candidates + selected:
        circular = normalize(item.get("circular"))
        if circular:
            gold.update(item["document_id"] for needle in needles if needle and needle in circular)
    return gold


def metric_for_case(case: dict, response: dict) -> dict:
    if not case.get("answerable", True):
        return {"id": case["id"], "answerable": False, "evaluation_status": "unanswerable_excluded",
                "department_group": "cross_department_or_unanswerable", "question_type": case.get("question_type"),
                "document_recall_at_30": None, "candidate_evidence_recall_at_30": None, "document_all_retrieved_at_30": None,
                "document_partial_coverage_at_30": None, "evidence_recall_at_6": None,
                "evidence_all_retrieved_at_6": None, "precision_at_6": None, "mrr": None,
                "fact_coverage": None, "gold_document_ids": [], "selected_evidence_ids": []}
    diagnostics = response.get("diagnostics")
    selected = response.get("evidence") or []
    if not isinstance(diagnostics, list):
        return {"id": case["id"], "answerable": True, "evaluation_status": "diagnostics_unavailable",
                "department_group": " / ".join(case.get("departments") or ["unknown"]), "question_type": case.get("question_type"),
                "document_recall_at_30": None, "candidate_evidence_recall_at_30": None, "document_all_retrieved_at_30": None,
                "document_partial_coverage_at_30": None, "evidence_recall_at_6": None,
                "evidence_all_retrieved_at_6": None, "precision_at_6": None, "mrr": None,
                "fact_coverage": None, "gold_document_ids": [], "selected_evidence_ids": [x.get("evidence_id") for x in selected]}
    candidates = sorted(diagnostics, key=lambda x: int(x.get("original_rank", 10**9)))
    candidates30 = candidates[:30]
    gold = gold_documents(case, candidates, selected)
    required_evidence = case.get("expected_evidence") or []
    candidate_evidence_hits = [spec for spec in required_evidence
                               if any(evidence_matches(item, spec) for item in candidates30)]
    evidence_hits = [spec for spec in required_evidence if any(evidence_matches(item, spec) for item in selected[:6])]
    evidence_by_id = {str(x.get("id")): x for x in required_evidence if x.get("id")}
    facts = [x for x in case.get("expected_facts") or [] if isinstance(x, dict) and x.get("evidence_ids")]
    fact_hits = sum(any(evidence_matches(item, evidence_by_id[eid]) and value_matches(str(fact.get("value", "")), item.get("text", ""), fact.get("type", "text"))
                     for eid in fact.get("evidence_ids", []) if eid in evidence_by_id for item in selected[:6]) for fact in facts)
    required_docs = gold
    candidate_docs = {x.get("document_id") for x in candidates30}
    first_rank = next((int(x["original_rank"]) for x in candidates30 if x.get("document_id") in required_docs), None)
    selected_relevant = sum(any(evidence_matches(item, spec) for spec in required_evidence)
                            for item in selected[:6]) if required_evidence else 0
    return {
        "id": case["id"], "answerable": True, "evaluation_status": "evaluated",
        "department_group": ("cross_department" if len(case.get("departments") or []) > 1
                              else (case.get("departments") or ["unknown"])[0]),
        "question_type": case.get("question_type"),
        "document_recall_at_30": len(required_docs & candidate_docs) / len(required_docs) if required_docs else None,
        "candidate_evidence_recall_at_30": len(candidate_evidence_hits) / len(required_evidence) if required_evidence else None,
        "document_all_retrieved_at_30": required_docs <= candidate_docs if required_docs else None,
        "document_partial_coverage_at_30": len(required_docs & candidate_docs) / len(required_docs) if required_docs else None,
        "evidence_recall_at_6": len(evidence_hits) / len(required_evidence) if required_evidence else None,
        "evidence_all_retrieved_at_6": len(evidence_hits) == len(required_evidence) if required_evidence else None,
        "precision_at_6": sum(any(evidence_matches(item, spec) for spec in required_evidence) for item in selected[:6]) / max(1, min(6, len(selected))) if required_evidence else None,
        "mrr": 1 / first_rank if first_rank else (0.0 if required_docs else None),
        "fact_coverage": fact_hits / len(facts) if facts else None,
        "required_document_count": len(required_docs), "candidate_document_hits": len(required_docs & candidate_docs),
        "required_evidence_count": len(required_evidence), "candidate_evidence_hits": len(candidate_evidence_hits),
        "selected_evidence_hits": selected_relevant, "selected_evidence_count": min(6, len(selected)),
        "facts_hit": fact_hits, "facts_expected": len(facts),
        "gold_document_ids": sorted(required_docs),
        "candidate_document_ids_at_30": sorted(candidate_docs),
        "selected_evidence_ids": [x.get("evidence_id") for x in selected],
    }


def answer_metrics(case: dict, response: dict, retrieval: dict, mode: str) -> dict:
    if mode != "end-to-end":
        return {"answer_correctness": None, "answer_fact_coverage": None, "citation_id_validity": None,
                "numerical_exact_match_accuracy": None, "citation_factual_support": None, "unsupported_claim_rate": None, "abstention_accuracy": None,
                "status": "unavailable_answer_generation_disabled"}
    answer = response.get("answer") or {}
    evidence_ids = {x.get("evidence_id") for x in response.get("evidence") or []}
    claims = answer.get("claims") or []
    unknown = [eid for claim in claims for eid in claim.get("evidence_ids", []) if eid not in evidence_ids]
    verified = [bool(claim.get("verified")) for claim in claims]
    answer_text = answer.get("answer", "")
    facts = [x for x in case.get("expected_facts") or [] if isinstance(x, dict)]
    answer_fact_hits = sum(value_matches(str(f.get("value", "")), answer_text, f.get("type", "text")) for f in facts)
    numeric_facts = [f for f in facts if f.get("type") == "number"]
    numeric_hits = sum(value_matches(str(f.get("value", "")), answer_text, "number") for f in numeric_facts)
    expected_status = "supported" if case.get("answerable", True) else "insufficient_evidence"
    return {"answer_correctness": None, "answer_fact_coverage": answer_fact_hits / len(facts) if facts else None,
            "numerical_exact_match_accuracy": numeric_hits / len(numeric_facts) if numeric_facts else None,
            "citation_id_validity": not unknown, "citation_factual_support": (all(verified) if claims else False),
            "unsupported_claim_rate": (sum(not x for x in verified) / len(verified)) if verified else 0.0,
            "abstention_accuracy": answer.get("answer_status") == expected_status,
            "answer_status": answer.get("answer_status"), "status": "deterministic_checks_only_answer_correctness_unavailable"}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values); index = (len(values) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (index - low)


def aggregate_group(entries: list[dict]) -> dict:
    def mean(key):
        values = [x[key] for x in entries if x.get(key) is not None]
        return statistics.mean(values) if values else None

    def ratio(numerator, denominator):
        total_n = sum(x.get(numerator, 0) or 0 for x in entries)
        total_d = sum(x.get(denominator, 0) or 0 for x in entries)
        return total_n / total_d if total_d else None

    return {
        "case_count": len({x.get("id") for x in entries}),
        "macro": {key: mean(key) for key in (
            "document_recall_at_30", "candidate_evidence_recall_at_30", "evidence_recall_at_6",
            "precision_at_6", "mrr", "fact_coverage")},
        "micro": {
            "document_recall_at_30": ratio("candidate_document_hits", "required_document_count"),
            "candidate_evidence_recall_at_30": ratio("candidate_evidence_hits", "required_evidence_count"),
            "evidence_recall_at_6": ratio("selected_evidence_hits", "required_evidence_count"),
            "fact_coverage": ratio("facts_hit", "facts_expected"),
        },
    }


def diagnostic_stages(case: dict, response: dict, retrieval: dict, answer: dict, mode: str) -> dict:
    diagnostics = response.get("diagnostics") or []
    required = case.get("expected_evidence") or []
    stage = {}
    for spec in required:
        matches = [item for item in diagnostics if evidence_matches(item, spec)]
        selected = [item for item in response.get("evidence") or [] if evidence_matches(item, spec)]
        if not matches:
            stage[spec.get("id")] = "initial_retrieval"
        elif not any(item.get("dense_retrieved") or item.get("lexical_retrieved") for item in matches):
            stage[spec.get("id")] = "fusion"
        elif not selected:
            top_rank = min((item.get("reranked_rank") or 10**9 for item in matches), default=10**9)
            stage[spec.get("id")] = "reranking_or_mmr" if top_rank > 6 else "mmr_selection"
        else:
            stage[spec.get("id")] = "answer_generation" if mode == "end-to-end" and case.get("answerable", True) and not any(
                spec.get("id") in claim.get("evidence_ids", []) and claim.get("verified")
                for claim in (answer.get("claims") or [])) else "retained"
    return {"case_id": case["id"], "question_type": case.get("question_type"),
            "department_group": retrieval.get("department_group"), "evidence_loss_stage": stage}


def summarize(results: list[dict], timings: list[dict], answer_results: list[dict]) -> dict:
    retrieval = {}
    for key in ("document_recall_at_30", "candidate_evidence_recall_at_30", "document_all_retrieved_at_30", "document_partial_coverage_at_30", "evidence_recall_at_6", "evidence_all_retrieved_at_6", "precision_at_6", "mrr", "fact_coverage"):
        values = [x[key] for x in results if x.get("answerable") and x.get(key) is not None]
        retrieval[key] = statistics.mean(values) if values else None
    latency = {}
    for key in ("embedding_ms", "qdrant_ms", "ranking_ms", "selection_ms", "gemini_ms", "retrieval_total_ms", "client_total_ms"):
        values = [float(x[key]) for x in timings if x.get(key) is not None]
        latency[key] = {"mean": statistics.mean(values) if values else None, "p50": percentile(values, .5), "p95": percentile(values, .95)}
    answer = {}
    for key in ("answer_fact_coverage", "numerical_exact_match_accuracy", "citation_id_validity", "citation_factual_support", "unsupported_claim_rate", "abstention_accuracy"):
        values = [x[key] for x in answer_results if x.get(key) is not None]
        answer[key] = statistics.mean(values) if values else None
    evaluated = [x for x in results if x.get("evaluation_status") == "evaluated"]
    groups = {}
    for field in ("department_group", "question_type"):
        groups[field] = {
            str(value): aggregate_group([x for x in evaluated if x.get(field) == value])
            for value in sorted({x.get(field) for x in evaluated if x.get(field)})
        }
    return {"retrieval_metrics": retrieval, "answer_metrics": answer,
            "latency_metrics": latency, "group_metrics": groups}


def markdown_report(report: dict) -> str:
    s = report["summary"]
    lines = [f"# FastAPI evaluation report: {report['mode']}", "", f"- API: `{report['base_url']}/api/v1/query`", f"- Cases: {report['case_count']}; successful requests: {report['successful_requests']}; failed requests: {report['failed_requests']}", f"- Warmup: {report['warmup_requested']}; measured runs per case: {report['runs_per_case']}", "", "## Effective configuration", "", "```json", json.dumps(report["configuration"], indent=2), "```", "", "## Metric definitions", "", "| Metric | Definition |", "|---|---|"]
    lines.extend(f"| {k} | {v} |" for k, v in METRIC_DEFINITIONS.items())
    lines += ["", "## Retrieval metrics", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {k} | {v if v is not None else 'unavailable'} |" for k, v in s["retrieval_metrics"].items())
    lines += ["", "## Answer metrics", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {k} | {v if v is not None else 'unavailable'} |" for k, v in s["answer_metrics"].items())
    lines += ["", "## Latency (ms)", "", "| Stage | Mean | p50 | p95 |", "|---|---:|---:|---:|"]
    for k, v in s["latency_metrics"].items():
        lines.append(f"| {k} | {v['mean'] if v['mean'] is not None else 'unavailable'} | {v['p50'] if v['p50'] is not None else 'unavailable'} | {v['p95'] if v['p95'] is not None else 'unavailable'} |")
    for group_name, group_values in s.get("group_metrics", {}).items():
        lines += ["", f"## Metrics by {group_name}", "", "| Group | Cases | Macro evidence recall@6 | Macro fact coverage | Micro evidence recall@6 |", "|---|---:|---:|---:|---:|"]
        for name, value in group_values.items():
            lines.append(f"| {name} | {value['case_count']} | {value['macro'].get('evidence_recall_at_6')} | {value['macro'].get('fact_coverage')} | {value['micro'].get('evidence_recall_at_6')} |")
    lines += ["", "## Limitations", "", "- Answer correctness is unavailable without an independent judge.", "- API failures are reported separately and are not converted into retrieval misses.", "- Retrieval metrics require diagnostics enabled on the FastAPI server."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("retrieval", "end-to-end"), required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--cases", type=Path, default=Path("evaluation/verified_cases_100.jsonl"))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-per-document", type=int, default=7)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-out", type=Path, default=Path("evaluation/fastapi_baseline_report.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path("evaluation/fastapi_baseline_report.md"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    cases = load_cases(args.cases)
    client = FastApiClient(args.base_url, args.timeout, args.retries)
    try:
        health = client.health()
    except Exception as exc:
        report = {"schema_version": "phase0-fastapi-evaluation-v1", "mode": args.mode,
                  "base_url": args.base_url.rstrip("/"), "health": None,
                  "configuration": {"candidate_k": args.candidate_k, "top_k": args.top_k,
                                     "max_per_document": args.max_per_document, "rerank": args.rerank},
                  "case_count": len(cases), "successful_requests": 0,
                  "failed_requests": len(cases), "warmup_requested": args.warmup,
                  "runs_per_case": args.runs, "cases": [], "answer_checks": [],
                  "failures": [{"stage": "health", "error": str(exc), "type": type(exc).__name__}],
                  "summary": {"retrieval_metrics": {}, "answer_metrics": {}, "latency_metrics": {}},
                  "limitations": ["FastAPI health check failed; no questions were evaluated."]}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        args.markdown_out.write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2
    results, answer_results, timings, failures, observed_parameters, diagnostics_report = [], [], [], [], [], []
    for case in cases:
        try:
            if args.warmup:
                client.query(case["question"], candidate_k=args.candidate_k, top_k=args.top_k, max_per_document=args.max_per_document, rerank=args.rerank, generate_answer=args.mode == "end-to-end", diagnostics=True)
            for run in range(args.runs):
                response = client.query(case["question"], candidate_k=args.candidate_k, top_k=args.top_k, max_per_document=args.max_per_document, rerank=args.rerank, generate_answer=args.mode == "end-to-end", diagnostics=True)
                observed_parameters.append(response.get("effective_parameters") or {})
                result = metric_for_case(case, response); result["run"] = run + 1
                answer_result = answer_metrics(case, response, result, args.mode)
                results.append(result); answer_results.append(answer_result)
                diagnostics_report.append(diagnostic_stages(case, response, result, response.get("answer") or {}, args.mode))
                timings.append({**(response.get("timings_ms") or {}), "client_total_ms": response.get("client_total_ms")})
        except Exception as exc:
            failures.append({"id": case["id"], "error": str(exc), "type": type(exc).__name__})
    observed = [x for x in results if x.get("evaluation_status") == "evaluated"]
    configuration = {"candidate_k": args.candidate_k, "top_k": args.top_k, "max_per_document": args.max_per_document, "rerank": args.rerank, "observed_response_parameters": sorted({json.dumps(x, sort_keys=True) for x in observed_parameters})}
    report = {"schema_version": "phase0-fastapi-evaluation-v2", "mode": args.mode, "base_url": args.base_url.rstrip("/"), "health": health, "configuration": configuration, "case_count": len(cases), "successful_requests": len(results), "failed_requests": len(failures), "warmup_requested": args.warmup, "runs_per_case": args.runs, "cases": results, "answer_checks": answer_results, "diagnostic_report": diagnostics_report, "failures": failures, "summary": summarize(observed, timings, answer_results), "limitations": ["Independent answer correctness judging was not run.", "Citation factual support is the backend's claim.verified result, not an independent judge.", "Reports are FastAPI-only and are not directly comparable to the historical CLI retrieval baseline."]}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown_out.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2)); print(f"Wrote {args.json_out} and {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
