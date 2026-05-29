"""Phase 2 acceptance tests for the SQL target.

Four things to verify:

  1. SqlTarget satisfies the Target protocol — its four sub-components
     implement the documented contracts. (Same pattern as the
     LongMemEvalTarget conformance test in Phase 1.)

  2. The SQL taxonomy classifies sample outcomes into the right
     regimes (one SqlOutcome per failure mode → one regime).

  3. The SQL action space gates a real prompt-transform end-to-end:
     static → compile → sandbox → install/revert.

  4. run_loop(target=SqlTarget(...)) completes on the mock fixture
     and produces the documented event-type sequence (loop.start →
     baseline → histogram → drafted → static → sandbox → eval_diff →
     promoted → attribution → ... → loop.stopped).
"""

from __future__ import annotations

import json
from pathlib import Path

from regimes.loop import run_loop
from regimes.target import ActionSpace, EvalBackend, RegimeTaxonomy, Target
from regimes.targets.sql import (
    FakeSqlReader,
    SqlActionSpace,
    SqlEvalBackend,
    SqlOutcome,
    SqlTarget,
    SqlTaxonomy,
    StubSqlAuthor,
    build_target,
)
from regimes.targets.sql import prompt_transforms as _pipeline


REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "synthetic_sql.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reader_table(data: list[dict]) -> dict[str, tuple[str, str, str]]:
    return {
        inst["question_id"]: (
            inst["gold_sql"], inst["default_wrong_sql"], inst["unlock_phrase"]
        )
        for inst in data
    }


def _make_outcome(**overrides) -> SqlOutcome:
    """Minimal SqlOutcome carrying schema + predicted/gold parse fields.
    Field defaults make every detector return False unless an override
    specifically trips it."""
    base = dict(
        question_id="q",
        question_type="simple-select",
        is_abstention=False,
        answer_session_ids=(),
        correct=False,
        gold_sql="SELECT name FROM customers;",
        predicted_sql="SELECT name FROM customers;",
        schema_tables=("customers",),
        schema_columns={"customers": ("id", "name")},
        gold_tables=("customers",),
        predicted_tables=("customers",),
    )
    base.update(overrides)
    return SqlOutcome(**base)


# ---------------------------------------------------------------------------
# 1) SqlTarget satisfies Target / sub-protocols
# ---------------------------------------------------------------------------


def test_sql_target_is_a_target():
    data = json.loads(FIXTURE.read_text())
    target = build_target(reader=FakeSqlReader(table=_reader_table(data)))
    assert isinstance(target, SqlTarget)
    assert isinstance(target, Target)
    assert isinstance(target.eval_backend, EvalBackend)
    assert isinstance(target.action_space, ActionSpace)
    assert isinstance(target.taxonomy, RegimeTaxonomy)
    assert target.name == "sql"


def test_action_space_and_taxonomy_independently_constructible():
    aspace = SqlActionSpace()
    tax = SqlTaxonomy()
    t = SqlTarget(
        eval_backend=SqlEvalBackend(reader=FakeSqlReader(table={})),
        action_space=aspace, taxonomy=tax,
    )
    assert isinstance(t, Target)


# ---------------------------------------------------------------------------
# 2) Taxonomy detectors
# ---------------------------------------------------------------------------


def test_classify_syntax_error_on_empty_sql():
    tax = SqlTaxonomy()
    o = _make_outcome(predicted_sql="")
    assert tax.classify(o).name == "syntax-error"


def test_classify_syntax_error_on_parse_failure():
    tax = SqlTaxonomy()
    o = _make_outcome(
        predicted_sql="SELECT name FORM customers;",
        exec_error='OperationalError: near "FORM": syntax error',
    )
    assert tax.classify(o).name == "syntax-error"


def test_classify_schema_misunderstanding_via_qualified_column():
    tax = SqlTaxonomy()
    o = _make_outcome(
        predicted_sql="SELECT customers.bogus FROM customers;",
        predicted_columns=(("customers", "bogus"),),
    )
    assert tax.classify(o).name == "schema-misunderstanding"


def test_classify_schema_misunderstanding_via_no_such_column():
    tax = SqlTaxonomy()
    o = _make_outcome(
        predicted_sql="SELECT title FROM customers;",
        exec_error="OperationalError: no such column: title",
    )
    assert tax.classify(o).name == "schema-misunderstanding"


def test_classify_wrong_aggregation_when_predicted_missing_group_by():
    tax = SqlTaxonomy()
    o = _make_outcome(
        gold_has_group_by=True,
        predicted_has_group_by=False,
    )
    assert tax.classify(o).name == "wrong-aggregation"


def test_classify_wrong_join_when_predicted_missing_join():
    tax = SqlTaxonomy()
    o = _make_outcome(
        gold_has_join=True,
        predicted_has_join=False,
    )
    assert tax.classify(o).name == "wrong-join"


def test_classify_wrong_filter_when_predicted_missing_where():
    tax = SqlTaxonomy()
    o = _make_outcome(
        gold_has_where=True,
        predicted_has_where=False,
    )
    assert tax.classify(o).name == "wrong-filter"


def test_classify_executable_but_wrong_is_the_wall():
    """Right structural shape, ran cleanly, wrong rows. The
    seam-unreachable failure — promotion can't fix it."""
    tax = SqlTaxonomy()
    o = _make_outcome(
        predicted_sql="SELECT name FROM customers WHERE country = 'NoSuch';",
        gold_has_where=True,
        predicted_has_where=True,
    )
    assert tax.classify(o).name == "executable-but-wrong"
    assert not tax.is_seam_reachable("executable-but-wrong")


def test_name_wall_describes_sql_specific_fixes():
    tax = SqlTaxonomy()
    wall = tax.name_wall({"executable-but-wrong": 3, "schema-misunderstanding": 1})
    assert "executable-but-wrong=3" in wall
    assert "SQL agent reasoning change" in wall
    # schema-misunderstanding is seam-reachable so doesn't appear in the wall.
    assert "schema-misunderstanding" not in wall


# ---------------------------------------------------------------------------
# 3) Action space gates a sample prompt-transform end-to-end
# ---------------------------------------------------------------------------


_GOOD_SOURCE = (
    "def transform(prompt_parts, question, schema_meta):\n"
    "    out = dict(prompt_parts)\n"
    "    hints = list(out.get('hints', []))\n"
    "    hints.append('be precise')\n"
    "    out['hints'] = hints\n"
    "    return out\n"
)


def test_action_space_static_gate_accepts_well_formed_transform():
    aspace = SqlActionSpace()
    res = aspace.static_gate(_GOOD_SOURCE)
    assert res.passed, res.reasons


def test_action_space_static_gate_rejects_wrong_signature():
    aspace = SqlActionSpace()
    res = aspace.static_gate(
        "def transform(scores, graph, question, question_date):\n"
        "    return scores\n"
    )
    assert not res.passed
    assert any("signature mismatch" in r for r in res.reasons)


def test_action_space_static_gate_rejects_disallowed_import():
    aspace = SqlActionSpace()
    res = aspace.static_gate(
        "import os\n"
        "def transform(prompt_parts, question, schema_meta):\n"
        "    return prompt_parts\n"
    )
    assert not res.passed
    assert any("os" in r for r in res.reasons)


def test_action_space_static_gate_allows_math_and_string():
    aspace = SqlActionSpace()
    res = aspace.static_gate(
        "import math\n"
        "import string\n"
        "def transform(prompt_parts, question, schema_meta):\n"
        "    _ = math.pi + len(string.ascii_letters)\n"
        "    return prompt_parts\n"
    )
    assert res.passed, res.reasons


def test_action_space_compile_and_sandbox():
    aspace = SqlActionSpace()
    fn = aspace.compile(_GOOD_SOURCE)
    probes = [{
        "prompt_parts": {"schema": "", "instructions": "", "hints": [], "question": ""},
        "question": "q",
        "schema_meta": {},
    }]
    res = aspace.sandbox_gate(fn, probes=probes)
    assert res.passed, res.reasons
    assert res.n_probed == 1


def test_action_space_sandbox_rejects_unknown_keys():
    """If the transform returns a dict with an extra key not in
    prompt_parts, sandbox flags it."""
    aspace = SqlActionSpace()
    fn = aspace.compile(
        "def transform(prompt_parts, question, schema_meta):\n"
        "    out = dict(prompt_parts)\n"
        "    out['bogus_new_key'] = 'x'\n"
        "    return out\n"
    )
    probes = [{
        "prompt_parts": {"schema": "", "hints": []},
        "question": "q",
        "schema_meta": {},
    }]
    res = aspace.sandbox_gate(fn, probes=probes)
    assert not res.passed
    assert any("unknown prompt_parts" in r for r in res.reasons)


def test_action_space_install_and_revert_round_trip():
    _pipeline.clear()
    try:
        aspace = SqlActionSpace()
        fn = aspace.compile(_GOOD_SOURCE)
        assert _pipeline.get_pipeline() == []
        aspace.install("t1", fn)
        assert [e.name for e in _pipeline.get_pipeline()] == ["t1"]
        aspace.revert("t1")
        assert _pipeline.get_pipeline() == []
    finally:
        _pipeline.clear()


# ---------------------------------------------------------------------------
# 4) run_loop(target=SqlTarget(...)) completes end-to-end
# ---------------------------------------------------------------------------


def test_run_loop_through_sql_target_completes():
    """The whole loop machinery runs against the SQL target with the
    documented semantic event sequence."""
    _pipeline.clear()
    try:
        data = json.loads(FIXTURE.read_text())
        target = build_target(
            reader=FakeSqlReader(table=_reader_table(data)),
            author=StubSqlAuthor(),
        )
        # Use the FULL fixture (not the split) so this test doesn't
        # depend on config/sql_split.json regenerating identically.
        rep = run_loop(
            target=target,
            instances=data[:12],          # small slice for speed
            pause_after=None,
            iteration_id="sql-test-run",
        )
        assert rep.baseline is not None
        assert rep.histogram is not None
        assert rep.stopped is not None

        # Filter to semantic events only.
        sem = [
            e.type for e in rep.events
            if e.type not in ("behavior.started", "behavior.completed", "runtime.idle")
        ]
        # First three must be loop.start, baseline.recorded, regime.histogram.
        assert sem[:3] == ["loop.start", "baseline.recorded", "regime.histogram"]
        # The chain MUST include at least one full transform-lifecycle
        # (drafted → static → sandbox → eval_diff → promoted-or-discarded).
        assert "transform.drafted" in sem
        assert "transform.static_passed" in sem
        assert "transform.sandbox_passed" in sem
        assert "transform.eval_diff" in sem
        assert "transform.promoted" in sem or "transform.discarded" in sem
        # Must end on loop.stopped.
        assert sem[-1] == "loop.stopped"
    finally:
        _pipeline.clear()


def test_run_loop_sql_baseline_in_documented_band():
    """Baseline accuracy on the full fixture should land in the
    50-75% band (the headroom we built in)."""
    _pipeline.clear()
    try:
        data = json.loads(FIXTURE.read_text())
        target = build_target(
            reader=FakeSqlReader(table=_reader_table(data)),
            author=StubSqlAuthor(),
        )
        result = target.eval_backend.run_on_split(data)
        acc = result.overall_accuracy()
        assert 0.50 <= acc <= 0.75, f"baseline {acc:.4f} outside 0.50-0.75"
    finally:
        _pipeline.clear()
