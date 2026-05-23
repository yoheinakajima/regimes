"""Tests for the frozen split + its invariants.

We do NOT regenerate split.json in tests; we verify the committed file
satisfies every invariant the loop will enforce at startup. The
generator itself is tested separately with a temp dir.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from activegraph import ConfigurationError

from regimes.split import load_split

REPO = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO / "config" / "split.json"


def test_committed_split_loads_clean() -> None:
    s = load_split(SPLIT_PATH)
    assert len(s.optimize) == 50
    assert len(s.confirm) == 100
    assert not (s.optimize_set & s.confirm_set)


def test_committed_split_required_types_in_both() -> None:
    s = load_split(SPLIT_PATH)
    # Verified inside load_split, but assert again from the public attrs.
    from regimes.split import REQUIRED_TYPES, _question_type
    for name, ids in (("optimize", s.optimize), ("confirm", s.confirm)):
        seen = {_question_type(q) for q in ids}
        assert REQUIRED_TYPES <= seen, f"{name} missing types"
        assert any(q.endswith("_abs") for q in ids), f"{name} has no _abs"


def test_loader_rejects_overlap(tmp_path: Path) -> None:
    base = json.loads(SPLIT_PATH.read_text())
    bad = dict(base)
    # plant the first confirm id into optimize
    bad["optimize"] = base["optimize"] + [base["confirm"][0]]
    p = tmp_path / "split.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ConfigurationError, match="overlap"):
        load_split(p)


def test_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="missing"):
        load_split(tmp_path / "nope.json")


def test_loader_rejects_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "split.json"
    p.write_text("{not json")
    with pytest.raises(ConfigurationError, match="not valid JSON"):
        load_split(p)


def test_loader_rejects_missing_abstention(tmp_path: Path) -> None:
    base = json.loads(SPLIT_PATH.read_text())
    bad = dict(base)
    # strip every _abs id from optimize
    bad["optimize"] = [q for q in base["optimize"] if not q.endswith("_abs")]
    p = tmp_path / "split.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ConfigurationError, match="_abs"):
        load_split(p)


def test_split_generator_is_deterministic(tmp_path: Path) -> None:
    """Re-running build_split on the committed fixture reproduces split.json byte-for-byte.

    Runs from REPO so the recorded source path stays relative ("fixtures/...").
    """
    out_rel = Path("artifact-split.json")
    res = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "build_split.py"),
            "--source",
            "fixtures/synthetic_lme.json",
            "--out",
            str(out_rel),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    try:
        assert res.returncode == 0, res.stderr
        regenerated = (REPO / out_rel).read_text()
        assert regenerated == SPLIT_PATH.read_text(), (
            "generator output drifted from committed split"
        )
    finally:
        (REPO / out_rel).unlink(missing_ok=True)


def test_fixture_generator_is_deterministic(tmp_path: Path) -> None:
    out = tmp_path / "synthetic_lme.json"
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_fixture.py"), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert out.read_text() == (REPO / "fixtures" / "synthetic_lme.json").read_text()
