"""Run the regimes loop against the SQL target.

Mirrors `scripts/run_loop.py` for LongMemEval, except the target is
`SqlTarget` and the eval backend is `SqlEvalBackend`.

Two modes:
  --mock   Run with FakeSqlReader (deterministic, no keys, no network).
           Each question's reader behavior is canned in
           `fixtures/synthetic_sql.json`: gold_sql vs. default_wrong_sql,
           plus an `unlock_phrase` a prompt-transform can inject to
           flip the question to correct. This is how the mock loop
           demonstrates promotion end-to-end.

  --real   Run with the Anthropic reader. Requires ANTHROPIC_API_KEY
           and the `regimes[eval]` extra. Not exercised in-container.

The loop machinery is unchanged from Phase 1; we just hand
`run_loop(target=SqlTarget(...))` instead of the legacy
`eval_backend=...+author=...` form."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from regimes.loop import run_loop  # noqa: E402
from regimes.split import load_split  # noqa: E402
from regimes.targets.sql import (  # noqa: E402
    FakeSqlReader,
    StubSqlAuthor,
    SqlActionSpace,
    SqlEvalBackend,
    SqlTarget,
    SqlTaxonomy,
)
from regimes.targets.sql import prompt_transforms as _pipeline  # noqa: E402


def _build_fake_reader_table(data: list[dict]) -> dict[str, tuple[str, str, str]]:
    """Map each question_id to (gold_sql, default_wrong_sql, unlock_phrase)
    so FakeSqlReader can answer deterministically."""
    return {
        inst["question_id"]: (
            inst["gold_sql"],
            inst["default_wrong_sql"],
            inst["unlock_phrase"],
        )
        for inst in data
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("mock", "real"), default="mock")
    p.add_argument("--full", action="store_true",
                   help="Run all phases through stop (default: pause after histogram).")
    p.add_argument("--run-dir", default="runs/sql_loop_001")
    p.add_argument("--fixture", default="fixtures/synthetic_sql.json")
    p.add_argument("--split", default="config/sql_split.json")
    args = p.parse_args()

    # Test isolation: any previously-installed prompt transforms from a
    # prior --mode mock run in the same process would carry over.
    _pipeline.clear()

    pause_after = None if args.full else "histogram"
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.fixture).read_text())
    s = load_split(args.split)
    by_id = {inst["question_id"]: inst for inst in data}
    optimize = [by_id[q] for q in s.optimize]
    confirm = [by_id[q] for q in s.confirm]

    if args.mode == "mock":
        reader = FakeSqlReader(table=_build_fake_reader_table(data))
        author = StubSqlAuthor()
    else:
        from regimes.eval import build_real_reader  # noqa: WPS433
        from regimes.targets.sql import build_real_sql_author  # noqa: WPS433
        reader = build_real_reader()
        author = build_real_sql_author()

    tax = SqlTaxonomy()
    action_space = SqlActionSpace(author=author, taxonomy=tax)
    target = SqlTarget(
        eval_backend=SqlEvalBackend(reader=reader),
        action_space=action_space,
        taxonomy=tax,
    )

    rep = run_loop(
        target=target,
        instances=optimize,
        confirm_instances=confirm,
        pause_after=pause_after,
        iteration_id="sql-loop-001",
    )

    # ----- print -----
    print()
    if rep.histogram is not None:
        print(rep.histogram["formatted"])
    print()
    if rep.baseline is not None:
        b = rep.baseline
        print(f"baseline overall_accuracy = {b['overall_accuracy']:.4f}")
        print("baseline per_type_accuracy:")
        for t in sorted(b["per_type_accuracy"]):
            print(f"  {t:30s}  {b['per_type_accuracy'][t]:.4f}")
        print()
    if rep.stopped is not None:
        print(f"stopped: {rep.stopped['reason']}")
        if rep.stopped.get("named_wall"):
            print(f"  wall: {rep.stopped['named_wall']}")
        if rep.stopped.get("remaining_regimes"):
            print(f"  remaining: {rep.stopped['remaining_regimes']}")
        print()
    if rep.transform_log:
        print("transform attempts (best-of-N audit):")
        for r in rep.transform_log:
            print(
                f"  {r['status']:18s} {r['name']:36s} "
                f"target={r['target_regime']:24s} "
                f"reasons={r.get('reasons', '')}"
            )
        print()

    payload = {
        "iteration_id": rep.iteration_id,
        "histogram": rep.histogram,
        "baseline": rep.baseline,
        "stopped": rep.stopped,
        "promotions": rep.promotions,
        "discards": rep.discards,
        "attributions": rep.attributions,
        "transform_log": rep.transform_log,
        "n_events": len(rep.events),
        "event_types": [e.type for e in rep.events],
    }
    out_path = run_dir / "report.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"report → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
