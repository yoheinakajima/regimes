"""Regression guard for Bug C — LMEJudge results-file parser.

Bug C symptom: the judge ran fine, wrote per-question verdicts to
`<hyp>.eval-results-gpt-4o`, stdout ended in "Accuracy: 0.78" with
clean per-type numbers — but the loop reported 0.0000 and put all 50
questions into assemble-internal.

Root cause: `_parse_per_question_results` did `splitlines()` +
`json.loads` per line, which silently empties the result when the
file is a JSON ARRAY (pretty-printed records). Even on the JSONL
path, records lack `question_id`, so the parser keyed every verdict
on None and they all collapsed into one slot. Downstream, RealEval
defaulted every qid to `{"correct": False, "label": "missing"}`.

The fix:
  1. Try whole-file `json.loads` first (handles the array/dict case)
     and fall back to JSONL.
  2. Map `autoeval_label` JSON booleans (true/false), legacy "0"/"1"
     strings, and "correct"/"wrong" strings to a single Python bool.
  3. When records don't carry `question_id`, join positionally to
     hypotheses.jsonl — upstream preserves input order.

These tests pin BOTH the parser's input-format tolerance and the
final per-type aggregate, so the all-False regression cannot recur.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from regimes.eval.real import _coerce_truthy_label, _parse_per_question_results


# ---------------------------------------------------------------------------
# Fixture: the on-disk shape upstream evaluate_qa.py writes (a JSON ARRAY
# of pretty-printed records; records carry `question/answer/hypothesis/
# autoeval_label` but NOT `question_id`). The values reproduce the
# user's reported run on the OPTIMIZE-50: overall 0.78 and per-type
# numbers single-session-assistant 1.0, knowledge-update 1.0,
# single-session-user 1.0, temporal-reasoning 6/14 = 0.4286,
# multi-session 11/13 = 0.8462, single-session-preference 2/3 = 0.6667.
# ---------------------------------------------------------------------------


PER_TYPE_BUILD: tuple[tuple[str, int, int], ...] = (
    # (question_type, n_total, n_correct). Numbers chosen so the
    # fixture sums to exactly 50 questions and 39 correct (overall
    # 0.78) with per-type ratios matching the user's reported run.
    ("single-session-assistant", 7, 7),    # 1.0
    ("knowledge-update", 7, 7),            # 1.0
    ("single-session-user", 6, 6),         # 1.0
    ("temporal-reasoning", 14, 6),         # 6/14 = 0.4286
    ("multi-session", 13, 11),             # 11/13 = 0.8462
    ("single-session-preference", 3, 2),   # 2/3 = 0.6667
    # totals: 50 questions, 39 correct → 0.78
)


def _build_results_array_fixture() -> tuple[list[dict], list[dict], dict]:
    """Returns (results_records, hypotheses_records, expected_per_type).

    Records are deliberately MISSING `question_id` to exercise the
    positional-join fallback — this is the upstream's actual shape."""
    results: list[dict] = []
    hyps: list[dict] = []
    expected: dict[str, float] = {}
    total = 0
    n_correct_total = 0
    for qtype, n_total, n_correct in PER_TYPE_BUILD:
        for i in range(n_total):
            qid = f"{qtype}_q{i:03d}"
            is_correct = i < n_correct
            # Upstream RECORD shape — no question_id; carries
            # question, answer, hypothesis, autoeval_label (bool).
            results.append({
                "question": f"q-{qid}",
                "answer": f"a-{qid}",
                "hypothesis": f"h-{qid}",
                "autoeval_label": is_correct,
            })
            # Hypotheses.jsonl: order preserved by upstream.
            hyps.append({"question_id": qid, "hypothesis": f"h-{qid}"})
            total += 1
            n_correct_total += int(is_correct)
        expected[qtype] = n_correct / n_total
    expected["__overall__"] = n_correct_total / total
    expected["__n__"] = total
    return results, hyps, expected


def _write_fixture(tmp_path: Path, results: list[dict], hyps: list[dict],
                   *, results_as_array: bool = True) -> tuple[Path, Path]:
    sub = tmp_path / "sub_1"
    sub.mkdir()
    hyp_path = sub / "hypotheses.jsonl"
    hyp_path.write_text("\n".join(json.dumps(h) for h in hyps) + "\n")
    res_path = sub / "hypotheses.jsonl.eval-results-gpt-4o"
    if results_as_array:
        # Pretty-printed JSON array — the upstream's actual format.
        res_path.write_text(json.dumps(results, indent=2) + "\n")
    else:
        # JSONL — the older format we still accept.
        res_path.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    return res_path, hyp_path


# ===========================================================================
# Parser format tolerance
# ===========================================================================


def test_parser_handles_pretty_printed_json_array(tmp_path):
    """The upstream's actual output shape. The pre-fix parser would
    splitlines() the array and json.loads each line — every line
    fails → empty result → all-False downstream."""
    results, hyps, _ = _build_results_array_fixture()
    res_path, hyp_path = _write_fixture(tmp_path, results, hyps,
                                        results_as_array=True)
    parsed = _parse_per_question_results(res_path, hyp_path=hyp_path)
    assert len(parsed) == len(results), (
        f"parser dropped records: got {len(parsed)}, expected {len(results)}"
    )


def test_parser_handles_jsonl_format(tmp_path):
    """Legacy format support — JSON-lines records, one per line."""
    results, hyps, _ = _build_results_array_fixture()
    res_path, hyp_path = _write_fixture(tmp_path, results, hyps,
                                        results_as_array=False)
    parsed = _parse_per_question_results(res_path, hyp_path=hyp_path)
    assert len(parsed) == len(results)


# ===========================================================================
# autoeval_label normalization
# ===========================================================================


@pytest.mark.parametrize("label,expected", [
    (True, True),
    (False, False),
    ("true", True),
    ("True", True),
    ("false", False),
    ("False", False),
    (1, True),
    (0, False),
    ("1", True),
    ("0", False),
    ("yes", True),
    ("no", False),
    ("correct", True),
    ("wrong", False),
    (None, False),
    ("", False),
    ("garbage", False),
])
def test_coerce_truthy_label_handles_all_known_shapes(label, expected):
    assert _coerce_truthy_label(label) is expected


# ===========================================================================
# Positional join (question_id missing on records)
# ===========================================================================


def test_parser_joins_by_position_when_question_id_absent(tmp_path):
    """Upstream records lack `question_id`; the parser pairs by
    position with hypotheses.jsonl so verdicts attach to the right
    instance. Without this fallback every verdict keys to None and
    they all collapse to one slot downstream."""
    results, hyps, _ = _build_results_array_fixture()
    res_path, hyp_path = _write_fixture(tmp_path, results, hyps,
                                        results_as_array=True)
    parsed = _parse_per_question_results(res_path, hyp_path=hyp_path)

    # Every record now carries a unique question_id from the hyp list.
    qids = [p["question_id"] for p in parsed]
    assert None not in qids, "positional join failed: at least one qid is None"
    assert len(set(qids)) == len(qids), (
        f"positional join produced duplicate qids: "
        f"{[q for q in qids if qids.count(q) > 1][:5]}"
    )
    # And they match the hypothesis order.
    expected = [h["question_id"] for h in hyps]
    assert qids == expected


def test_parser_prefers_embedded_question_id_when_present(tmp_path):
    """If upstream ever starts emitting question_id, the parser must
    trust it over the positional fallback (in case of ordering drift)."""
    results, hyps, _ = _build_results_array_fixture()
    # Inject an explicit question_id that DIFFERS from the positional
    # match, and confirm the embedded id wins.
    results[0]["question_id"] = "EXPLICIT_FIRST"
    res_path, hyp_path = _write_fixture(tmp_path, results, hyps,
                                        results_as_array=True)
    parsed = _parse_per_question_results(res_path, hyp_path=hyp_path)
    assert parsed[0]["question_id"] == "EXPLICIT_FIRST"


# ===========================================================================
# End-to-end: parsed verdicts → aggregate matches upstream Accuracy: 0.78
# ===========================================================================


def test_parsed_aggregate_reproduces_judge_accuracy_and_per_type(tmp_path):
    """The headline regression guard: fixture replicates the user's
    observed run (overall 0.78, per-type numbers as reported in the
    bug). After parsing, computing the aggregate the same way RealEval
    does, we must reproduce those numbers — NOT 0.0 with all questions
    bucketed wrong."""
    results, hyps, expected = _build_results_array_fixture()
    res_path, hyp_path = _write_fixture(tmp_path, results, hyps,
                                        results_as_array=True)
    parsed = _parse_per_question_results(res_path, hyp_path=hyp_path)

    # Map qid → question_type from the original build for the aggregate.
    qid_to_type: dict[str, str] = {}
    for r in PER_TYPE_BUILD:
        qtype, n_total, _ = r
        for i in range(n_total):
            qid_to_type[f"{qtype}_q{i:03d}"] = qtype

    correct_by_type: dict[str, int] = defaultdict(int)
    total_by_type: dict[str, int] = defaultdict(int)
    n_correct = 0
    for p in parsed:
        qt = qid_to_type[p["question_id"]]
        total_by_type[qt] += 1
        if p["correct"]:
            correct_by_type[qt] += 1
            n_correct += 1
    overall = n_correct / len(parsed)
    per_type = {qt: correct_by_type[qt] / total_by_type[qt]
                for qt in sorted(total_by_type)}

    assert overall == pytest.approx(expected["__overall__"], abs=1e-9)
    assert overall == pytest.approx(0.78, abs=0.02), (
        f"overall accuracy {overall} doesn't match the user's reported "
        f"0.78 — Bug C regression."
    )
    for qt, want in expected.items():
        if qt.startswith("__"):
            continue
        assert per_type[qt] == pytest.approx(want, abs=1e-6), (
            f"per_type[{qt}] = {per_type[qt]} != {want}"
        )


# ===========================================================================
# Make-it-impossible-to-silently-regress: empty input file
# ===========================================================================


def test_parser_returns_empty_list_on_truly_empty_results_file(tmp_path):
    """The fix must NOT introduce a different failure mode: if the
    file really IS empty / unparseable, return an empty list (the
    previous behavior) — so the downstream "all defaults to missing"
    is visible rather than silently inventing verdicts."""
    res_path = tmp_path / "empty.eval-results-gpt-4o"
    res_path.write_text("")
    parsed = _parse_per_question_results(res_path)
    assert parsed == []


def test_parser_handles_object_with_results_key(tmp_path):
    """Some upstream variants wrap the array under a top-level key."""
    results, hyps, _ = _build_results_array_fixture()
    sub = tmp_path / "sub_1"
    sub.mkdir()
    hyp_path = sub / "hypotheses.jsonl"
    hyp_path.write_text("\n".join(json.dumps(h) for h in hyps) + "\n")
    res_path = sub / "hypotheses.jsonl.eval-results-gpt-4o"
    res_path.write_text(json.dumps({"results": results}, indent=2) + "\n")
    parsed = _parse_per_question_results(res_path, hyp_path=hyp_path)
    assert len(parsed) == len(results)


# ===========================================================================
# Symptom-level guard: diagnose must NOT collapse all failures into one
# bucket when verdicts are real (which is what 0.0 + 50× assemble-internal
# was — a downstream symptom of the parser's all-False output).
# ===========================================================================


def _outcome_for(qid, qtype, *, correct, selected, gold_sids, scores=None,
                 ranked=None, truncated=False, decisions=()):
    from regimes.eval.types import Outcome
    return Outcome(
        question_id=qid, question_type=qtype, is_abstention=False,
        answer_session_ids=tuple(gold_sids),
        correct=correct,
        selected_turn_ids=tuple(selected),
        scores=scores or {},
        ranked=tuple(ranked or ()),
        truncated=truncated,
        decisions=tuple(decisions),
    )


def test_diagnose_does_not_collapse_failures_to_assemble_internal_when_real():
    """When verdicts are real (mix of correct/wrong) AND outcomes carry
    varied retrieval shapes (some gold selected, some not, some
    truncated at budget, some scoring-error), the histogram must
    distribute failures across regimes.

    The original bug presented as 50/50 in assemble-internal — that's
    only possible when EVERY failing outcome happens to have gold in
    selected_turn_ids, which itself only happens when verdicts come
    back uniformly False on a corpus where the agent reliably retrieves
    gold. That uniformity was the all-False parser symptom; with real
    verdicts the distribution must be non-trivial."""
    from regimes.loop.regimes import histogram as regime_histogram

    outs = [
        # multi-session, gold selected and correct
        _outcome_for("q01", "multi-session", correct=True,
                    selected=["s1#0"], gold_sids=["s1"],
                    scores={"s1#0": 0.9}, ranked=["s1#0"]),
        # temporal, gold well-ranked and mostly selected, but wrong →
        # assemble-internal (coverage-based detector requires gold in
        # top-K to be present and >= floor selected).
        _outcome_for("q02", "temporal-reasoning", correct=False,
                    selected=["s2#0"], gold_sids=["s2"],
                    scores={"s2#0": 0.9}, ranked=["s2#0"]),
        _outcome_for("q03", "temporal-reasoning", correct=False,
                    selected=["s3#0"], gold_sids=["s3"],
                    scores={"s3#0": 0.85}, ranked=["s3#0"]),
        # temporal, gold ranked top-5 but not selected → assembly-crowding
        _outcome_for("q04", "temporal-reasoning", correct=False,
                    selected=["other#0"], gold_sids=["s4"],
                    scores={"s4#0": 0.5, "other#0": 0.9},
                    ranked=["other#0", "s4#0"],
                    truncated=True),
        # multi-session, gold dropped at budget → budget-truncation
        _outcome_for("q05", "multi-session", correct=False,
                    selected=["other#0"], gold_sids=["s5"],
                    scores={"s5#0": 0.7, "other#0": 0.9},
                    ranked=["other#0", "s5#0"],
                    truncated=True,
                    decisions=[{"turn_id": "s5#0", "included": False,
                                "reason": "budget"}]),
        # preference, scoring error
        _outcome_for("q06", "single-session-preference", correct=False,
                    selected=[], gold_sids=["s6"],
                    scores={}),
    ]
    # The scoring-error signal in the current code reads from
    # Outcome.score_error; emulate that explicitly.
    from dataclasses import replace
    outs[5] = replace(outs[5], score_error="agent.score_embedding:BadRequest")

    rows = regime_histogram(outs)
    by_regime = {r.regime: r.count for r in rows}
    # 5 failures spread across multiple regimes — NOT all in one bucket.
    assert by_regime["assemble-internal"] == 2          # q02, q03
    assert by_regime["assembly-crowding"] == 1          # q04
    assert by_regime["budget-truncation"] == 1          # q05
    assert by_regime["scoring-error"] == 1              # q06
    # Sanity: nothing collapsed to one bucket.
    distinct = sum(1 for c in by_regime.values() if c > 0)
    assert distinct >= 3, (
        f"failures collapsed into too few buckets: {by_regime} — "
        f"this is the shape of the original Bug C symptom."
    )
