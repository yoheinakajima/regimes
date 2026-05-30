"""Tests for the widened LongMemEval action space: three transform types.

Verifies:
  1. Per-type static-gate: malformed transforms rejected with clear reason;
     valid ones pass.
  2. Assembly-transform no-fabricated-ids invariant enforced.
  3. Reader-prompt-transform length-cap enforced.
  4. Selective-drafting routes regime → correct type.
  5. Mock loop exercises all three types through the gate chain.
"""

from __future__ import annotations

import pytest

from regimes.loop.gates import compile_transform, static_gate
from regimes.loop.mock_eval import MockEval, MockInstance
from regimes.targets.longmemeval.action_space import (
    LongMemEvalActionSpace,
    clear_all_pipelines,
)
from regimes.targets.longmemeval.mock_author import MockTypedAuthor
from regimes.targets.longmemeval.transform_types import (
    ALL_TRANSFORM_TYPES,
    ASSEMBLY_TRANSFORM,
    READER_PROMPT_MAX_ADDED_CHARS,
    READER_PROMPT_TRANSFORM,
    REGIME_TO_TYPES,
    SCORE_TRANSFORM,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_all_pipelines()
    yield
    clear_all_pipelines()


# ===========================================================================
# 1. Per-type static-gate tests
# ===========================================================================


class TestScoreTransformStaticGate:
    VALID = (
        "def transform(scores, graph, question, question_date):\n"
        "    return {t: s * 1.1 for t, s in scores.items()}\n"
    )

    def test_valid_passes(self):
        t = SCORE_TRANSFORM
        r = static_gate(self.VALID, signature_params=t.signature_params,
                        import_whitelist=t.import_whitelist)
        assert r.passed, r.reasons

    def test_wrong_signature_rejected(self):
        src = "def transform(x):\n    return x\n"
        t = SCORE_TRANSFORM
        r = static_gate(src, signature_params=t.signature_params,
                        import_whitelist=t.import_whitelist)
        assert not r.passed
        assert any("signature mismatch" in x for x in r.reasons)

    def test_banned_import_rejected(self):
        src = "import os\n" + self.VALID
        t = SCORE_TRANSFORM
        r = static_gate(src, signature_params=t.signature_params,
                        import_whitelist=t.import_whitelist)
        assert not r.passed
        assert any("import outside whitelist" in x for x in r.reasons)


class TestAssemblyTransformStaticGate:
    VALID = (
        "def transform(selected_turns, scores, question, question_date):\n"
        "    return sorted(selected_turns, key=lambda t: -scores.get(t, 0.0))\n"
    )

    def test_valid_passes(self):
        t = ASSEMBLY_TRANSFORM
        r = static_gate(self.VALID, signature_params=t.signature_params,
                        import_whitelist=t.import_whitelist)
        assert r.passed, r.reasons

    def test_wrong_signature_rejected(self):
        src = "def transform(turns):\n    return turns\n"
        t = ASSEMBLY_TRANSFORM
        r = static_gate(src, signature_params=t.signature_params,
                        import_whitelist=t.import_whitelist)
        assert not r.passed
        assert any("signature mismatch" in x for x in r.reasons)

    def test_os_import_rejected(self):
        src = "import os\n" + self.VALID
        t = ASSEMBLY_TRANSFORM
        r = static_gate(src, signature_params=t.signature_params,
                        import_whitelist=t.import_whitelist)
        assert not r.passed
        assert any("import outside whitelist" in x for x in r.reasons)

    def test_string_import_allowed(self):
        src = "import string\n" + self.VALID
        t = ASSEMBLY_TRANSFORM
        r = static_gate(src, signature_params=t.signature_params,
                        import_whitelist=t.import_whitelist)
        assert r.passed, r.reasons


class TestReaderPromptTransformStaticGate:
    VALID = (
        "def transform(prompt_parts, question, question_date):\n"
        "    out = dict(prompt_parts)\n"
        "    out['instruction'] = out.get('instruction', '') + ' Be precise.'\n"
        "    return out\n"
    )

    def test_valid_passes(self):
        t = READER_PROMPT_TRANSFORM
        r = static_gate(self.VALID, signature_params=t.signature_params,
                        import_whitelist=t.import_whitelist)
        assert r.passed, r.reasons

    def test_wrong_signature_rejected(self):
        src = "def transform(parts):\n    return parts\n"
        t = READER_PROMPT_TRANSFORM
        r = static_gate(src, signature_params=t.signature_params,
                        import_whitelist=t.import_whitelist)
        assert not r.passed
        assert any("signature mismatch" in x for x in r.reasons)

    def test_network_import_rejected(self):
        src = "import socket\n" + self.VALID
        t = READER_PROMPT_TRANSFORM
        r = static_gate(src, signature_params=t.signature_params,
                        import_whitelist=t.import_whitelist)
        assert not r.passed
        assert any("import outside whitelist" in x for x in r.reasons)


# ===========================================================================
# 2. Assembly-transform: no-fabricated-ids invariant
# ===========================================================================


class TestAssemblyNoFabricatedIds:
    def test_subset_passes(self):
        src = (
            "def transform(selected_turns, scores, question, question_date):\n"
            "    return selected_turns[:2]\n"
        )
        fn = compile_transform(src)
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        aspace._active_type = ASSEMBLY_TRANSFORM
        probes = [{
            "selected_turns": ["t1", "t2", "t3"],
            "scores": {"t1": 1.0, "t2": 0.5, "t3": 0.3},
            "question": "q",
            "question_date": "",
        }]
        r = aspace._sandbox_with_probe_context(fn, probes, ASSEMBLY_TRANSFORM)
        assert r.passed, r.reasons

    def test_fabricated_id_rejected(self):
        src = (
            "def transform(selected_turns, scores, question, question_date):\n"
            "    return selected_turns + ['FABRICATED_ID']\n"
        )
        fn = compile_transform(src)
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        aspace._active_type = ASSEMBLY_TRANSFORM
        probes = [{
            "selected_turns": ["t1", "t2"],
            "scores": {"t1": 1.0, "t2": 0.5},
            "question": "q",
            "question_date": "",
        }]
        r = aspace._sandbox_with_probe_context(fn, probes, ASSEMBLY_TRANSFORM)
        assert not r.passed
        assert any("fabricated" in x for x in r.reasons)

    def test_reorder_passes(self):
        src = (
            "def transform(selected_turns, scores, question, question_date):\n"
            "    return list(reversed(selected_turns))\n"
        )
        fn = compile_transform(src)
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        aspace._active_type = ASSEMBLY_TRANSFORM
        probes = [{
            "selected_turns": ["t1", "t2", "t3"],
            "scores": {"t1": 1.0, "t2": 0.5, "t3": 0.3},
            "question": "q",
            "question_date": "",
        }]
        r = aspace._sandbox_with_probe_context(fn, probes, ASSEMBLY_TRANSFORM)
        assert r.passed, r.reasons


# ===========================================================================
# 3. Reader-prompt-transform: length cap + same-keys invariant
# ===========================================================================


class TestReaderPromptLengthCap:
    def test_small_addition_passes(self):
        src = (
            "def transform(prompt_parts, question, question_date):\n"
            "    out = dict(prompt_parts)\n"
            "    out['instruction'] = out.get('instruction', '') + ' Be precise.'\n"
            "    return out\n"
        )
        fn = compile_transform(src)
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        aspace._active_type = READER_PROMPT_TRANSFORM
        probes = [{
            "prompt_parts": {"context": "some context", "instruction": "answer"},
            "question": "q",
            "question_date": "",
        }]
        r = aspace._sandbox_with_probe_context(fn, probes, READER_PROMPT_TRANSFORM)
        assert r.passed, r.reasons

    def test_exceeds_length_cap_rejected(self):
        big_addition = "x" * (READER_PROMPT_MAX_ADDED_CHARS + 100)
        src = (
            "def transform(prompt_parts, question, question_date):\n"
            "    out = dict(prompt_parts)\n"
            f"    out['instruction'] = out.get('instruction', '') + '{big_addition}'\n"
            "    return out\n"
        )
        fn = compile_transform(src)
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        aspace._active_type = READER_PROMPT_TRANSFORM
        probes = [{
            "prompt_parts": {"context": "some context", "instruction": "answer"},
            "question": "q",
            "question_date": "",
        }]
        r = aspace._sandbox_with_probe_context(fn, probes, READER_PROMPT_TRANSFORM)
        assert not r.passed
        assert any("exceeds cap" in x for x in r.reasons)

    def test_fabricated_key_rejected(self):
        src = (
            "def transform(prompt_parts, question, question_date):\n"
            "    out = dict(prompt_parts)\n"
            "    out['new_key'] = 'smuggled'\n"
            "    return out\n"
        )
        fn = compile_transform(src)
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        aspace._active_type = READER_PROMPT_TRANSFORM
        probes = [{
            "prompt_parts": {"context": "some context", "instruction": "answer"},
            "question": "q",
            "question_date": "",
        }]
        r = aspace._sandbox_with_probe_context(fn, probes, READER_PROMPT_TRANSFORM)
        assert not r.passed
        assert any("unknown keys" in x or "fabricated" in x for x in r.reasons)

    def test_missing_key_rejected(self):
        src = (
            "def transform(prompt_parts, question, question_date):\n"
            "    return {'instruction': prompt_parts.get('instruction', '')}\n"
        )
        fn = compile_transform(src)
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        aspace._active_type = READER_PROMPT_TRANSFORM
        probes = [{
            "prompt_parts": {"context": "some context", "instruction": "answer"},
            "question": "q",
            "question_date": "",
        }]
        r = aspace._sandbox_with_probe_context(fn, probes, READER_PROMPT_TRANSFORM)
        assert not r.passed
        assert any("missing keys" in x or "same keys" in x for x in r.reasons)


# ===========================================================================
# 4. Selective drafting: regime → type routing
# ===========================================================================


class TestSelectiveDrafting:
    def test_budget_truncation_selects_score_transform(self):
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        t = aspace._select_type("budget-truncation")
        assert t.name in ("score-transform", "assembly-transform")

    def test_assembly_crowding_selects_score_or_assembly(self):
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        t = aspace._select_type("assembly-crowding")
        assert t.name in ("score-transform", "assembly-transform")

    def test_assemble_internal_selects_reader_prompt(self):
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        t = aspace._select_type("assemble-internal")
        assert t.name == "reader-prompt-transform"

    def test_retrieval_signal_gap_falls_back_to_score(self):
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        t = aspace._select_type("retrieval-signal-gap")
        # Wall regime — no eligible types, falls back to score-transform
        assert t.name == "score-transform"

    def test_draft_sets_active_type(self):
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        from regimes.eval.types import Outcome
        aspace.draft(dominant_regime="assemble-internal", failures=[])
        assert aspace._active_type.name == "reader-prompt-transform"

    def test_draft_for_budget_truncation(self):
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        change = aspace.draft(dominant_regime="budget-truncation", failures=[])
        assert change.target_regime == "budget-truncation"
        assert "def transform" in change.source

    def test_draft_for_assemble_internal(self):
        aspace = LongMemEvalActionSpace(author=MockTypedAuthor())
        change = aspace.draft(dominant_regime="assemble-internal", failures=[])
        assert change.target_regime == "assemble-internal"
        assert "prompt_parts" in change.source


# ===========================================================================
# 5. Full gate chain for each type (mock loop)
# ===========================================================================


class TestFullGateChainScoreTransform:
    """Score-transform type flows through all gates."""

    def test_score_transform_gates_pass(self):
        author = MockTypedAuthor()
        aspace = LongMemEvalActionSpace(author=author)
        # Draft for assembly-crowding → score-transform
        change = aspace.draft(dominant_regime="assembly-crowding", failures=[])
        # Static gate
        static = aspace.static_gate(change.source)
        assert static.passed, static.reasons
        # Compile
        fn = aspace.compile(change.source)
        assert callable(fn)
        # Build probes
        ev = MockEval()
        instances = [
            MockInstance("q1", "multi-session", False, ("s1",), True,
                         scores={"s1#0": 1.0, "s2#0": 0.5},
                         selected_turn_ids=("s1#0", "s2#0")),
        ]
        baseline = ev.run_on_split(instances)
        probes = aspace.build_probes(baseline)
        assert len(probes) > 0
        # Sandbox gate
        sandbox = aspace.sandbox_gate(fn, probes=probes)
        assert sandbox.passed, sandbox.reasons


class TestFullGateChainAssemblyTransform:
    """Assembly-transform type flows through all gates."""

    def test_assembly_transform_gates_pass(self):
        author = MockTypedAuthor()
        aspace = LongMemEvalActionSpace(author=author)
        # Force assembly-transform type
        aspace._active_type = ASSEMBLY_TRANSFORM
        change = author.draft_typed(
            dominant_regime="assembly-crowding",
            failures=[],
            transform_type="assembly-transform",
        )
        # Static gate
        static = aspace.static_gate(change.source)
        assert static.passed, static.reasons
        # Compile
        fn = aspace.compile(change.source)
        assert callable(fn)
        # Build probes
        ev = MockEval()
        instances = [
            MockInstance("q1", "multi-session", False, ("s1",), True,
                         scores={"s1#0": 1.0, "s2#0": 0.5},
                         selected_turn_ids=("s1#0", "s2#0")),
        ]
        baseline = ev.run_on_split(instances)
        probes = aspace.build_probes(baseline)
        assert len(probes) > 0
        assert "selected_turns" in probes[0]
        # Sandbox gate
        sandbox = aspace.sandbox_gate(fn, probes=probes)
        assert sandbox.passed, sandbox.reasons


class TestFullGateChainReaderPromptTransform:
    """Reader-prompt-transform type flows through all gates."""

    def test_reader_prompt_transform_gates_pass(self):
        author = MockTypedAuthor()
        aspace = LongMemEvalActionSpace(author=author)
        # Draft for assemble-internal → reader-prompt-transform
        change = aspace.draft(dominant_regime="assemble-internal", failures=[])
        assert aspace._active_type.name == "reader-prompt-transform"
        # Static gate
        static = aspace.static_gate(change.source)
        assert static.passed, static.reasons
        # Compile
        fn = aspace.compile(change.source)
        assert callable(fn)
        # Build probes
        ev = MockEval()
        instances = [
            MockInstance("q1", "multi-session", False, ("s1",), True,
                         scores={"s1#0": 1.0},
                         selected_turn_ids=("s1#0",)),
        ]
        baseline = ev.run_on_split(instances)
        probes = aspace.build_probes(baseline)
        assert len(probes) > 0
        assert "prompt_parts" in probes[0]
        # Sandbox gate
        sandbox = aspace.sandbox_gate(fn, probes=probes)
        assert sandbox.passed, sandbox.reasons


# ===========================================================================
# 6. End-to-end: mock loop with MockTypedAuthor exercises all three types
# ===========================================================================


class TestMockLoopAllTypes:
    """The mock loop runs with each transform type being drafted and gated."""

    def test_mock_loop_score_transform(self):
        """Score-transform path still works through the full loop."""
        from regimes.loop import run_loop, TRANSFORM_DRAFTED, TRANSFORM_STATIC_PASSED

        author = MockTypedAuthor()
        instances = [
            MockInstance("q_ok", "multi-session", False, ("s1",), True,
                         scores={"s1#0": 1.0}, selected_turn_ids=("s1#0",)),
            MockInstance("q_ac", "multi-session", False, ("sG",), False,
                         scores={"sG#0": 0.6, "sN#0": 0.9},
                         ranked=("sN#0", "sG#0"),
                         selected_turn_ids=("sN#0",), truncated=True),
        ]
        rep = run_loop(
            eval_backend=MockEval(),
            instances=instances,
            author=author,
            max_consecutive_discards=1,
        )
        types_seen = {e.type for e in rep.events}
        assert TRANSFORM_DRAFTED in types_seen
        assert TRANSFORM_STATIC_PASSED in types_seen

    def test_mock_loop_assembly_transform(self):
        """Assembly-transform can be drafted and pass static+sandbox gates."""
        author = MockTypedAuthor()
        aspace = LongMemEvalActionSpace(author=author)

        # Directly exercise draft → static → compile → sandbox
        aspace._active_type = ASSEMBLY_TRANSFORM
        change = author.draft_typed(
            dominant_regime="budget-truncation",
            failures=[],
            transform_type="assembly-transform",
        )
        static = aspace.static_gate(change.source)
        assert static.passed, static.reasons
        fn = aspace.compile(change.source)
        probes = [{
            "selected_turns": ["t1", "t2", "t3"],
            "scores": {"t1": 0.9, "t2": 0.5, "t3": 0.3},
            "question": "",
            "question_date": "",
        }]
        sandbox = aspace.sandbox_gate(fn, probes=probes)
        assert sandbox.passed, sandbox.reasons

    def test_mock_loop_reader_prompt_transform(self):
        """Reader-prompt-transform can be drafted and pass static+sandbox."""
        author = MockTypedAuthor()
        aspace = LongMemEvalActionSpace(author=author)

        change = aspace.draft(dominant_regime="assemble-internal", failures=[])
        assert aspace._active_type.name == "reader-prompt-transform"
        static = aspace.static_gate(change.source)
        assert static.passed, static.reasons
        fn = aspace.compile(change.source)
        probes = [{
            "prompt_parts": {"context": "ctx", "instruction": "answer"},
            "question": "",
            "question_date": "",
        }]
        sandbox = aspace.sandbox_gate(fn, probes=probes)
        assert sandbox.passed, sandbox.reasons


# ===========================================================================
# 7. Confirm threshold is configurable on the action space
# ===========================================================================


class TestConfirmThreshold:
    def test_default_threshold_is_zero(self):
        aspace = LongMemEvalActionSpace()
        assert aspace.confirm_threshold == 0.0

    def test_threshold_is_configurable(self):
        aspace = LongMemEvalActionSpace(confirm_threshold=0.02)
        assert aspace.confirm_threshold == 0.02
