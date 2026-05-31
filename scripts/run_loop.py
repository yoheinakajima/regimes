"""Run the regimes loop and report the regime histogram.

Two modes:

  --mock      Run against the in-repo synthetic fixture using MockEval.
              No keys, no network. Used for CI + design demos. Produces
              an EvalResult derived from the synthetic instances; each
              MockInstance can be configured for a specific regime.

  --real      Run against the real LME data + AnthropicReader + LMEJudge.
              Requires: ANTHROPIC_API_KEY, OPENAI_API_KEY, the LME
              checkout (`activegraph-longmemeval/`), and the
              `regimes[eval]` extra installed.

By default we PAUSE at the histogram: the histogram is the go/no-go
checkpoint before spending eval budget on transforms. Use
`--full` to run all phases through stop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from regimes.loop import MockEval, MockInstance, run_loop  # noqa: E402
from regimes.loop.hypothesize import build_real_author  # noqa: E402
from regimes.split import load_split  # noqa: E402


def _build_mock_confirm_instances() -> list[MockInstance]:
    """A small held-out CONFIRM fixture used in --mode mock to exercise
    the loop's CONFIRM gate. Disjoint from OPTIMIZE. One instance is
    flippable by stub_topk_boost so confirm_delta is a non-zero number
    in the mock report — proof that confirm_instances are threaded all
    the way through to the promotion gate."""
    return [
        MockInstance("qc_ok1", "multi-session", False, ("sc1",), True,
                     scores={"sc1#0": 1.0}, selected_turn_ids=("sc1#0",)),
        MockInstance("qc_ok2", "knowledge-update", False, ("sc2",), True,
                     scores={"sc2#0": 0.92}, selected_turn_ids=("sc2#0",)),
        MockInstance(
            "qc_ac1", "multi-session", False, ("sGc_ac",), False,
            scores={"sGc_ac#0": 0.6, "sNc#0": 0.95},
            ranked=("sNc#0", "sGc_ac#0"),
            selected_turn_ids=("sNc#0",), truncated=True,
            gold_score_threshold=0.7,
            candidate_turn_ids=("sGc_ac#0",),
        ),
    ]


def _build_chaotic_mock_instances() -> list[MockInstance]:
    """A budget-truncation-dominant + assemble-internal fixture that
    forces the loop to ROTATE between two seam-reachable regimes under a
    chaotic author. Mirrors the real run's baseline shape: drafting
    score-transforms for budget-truncation (which the chaotic author
    botches), then rotating to assemble-internal and drafting a
    reader-prompt-transform that CAN promote (the assemble-internal
    instances carry prompt fragments + the reconciliation marker)."""
    from regimes.targets.longmemeval.mock_author import RECONCILE_MARKER

    insts: list[MockInstance] = [
        MockInstance("q_ok", "multi-session", False, ("s_ok",), True,
                     scores={"s_ok#0": 1.0}, selected_turn_ids=("s_ok#0",)),
    ]
    # 4 budget-truncation: gold dropped at the budget wall, no flip path.
    for i in range(4):
        insts.append(MockInstance(
            f"q_bt{i}", "multi-session", False, (f"sGb{i}",), False,
            scores={f"sGb{i}#0": 0.7, f"sNb{i}#0": 0.6},
            ranked=(f"sGb{i}#0", f"sNb{i}#0"),
            selected_turn_ids=(f"sNb{i}#0",), truncated=True,
            decisions=({"turn_id": f"sGb{i}#0", "included": False, "reason": "budget"},),
            candidate_turn_ids=(f"sGb{i}#0", f"sNb{i}#0"),
        ))
    # 3 assemble-internal: gold fully selected, answer wrong, not truncated;
    # a reader-prompt-transform injecting the marker flips them correct.
    for i in range(3):
        gold = f"sGa{i}"
        insts.append(MockInstance(
            f"q_ai{i}", "multi-session", False, (gold,), False,
            scores={f"{gold}#0": 0.9, f"sNa{i}#0": 0.4},
            ranked=(f"{gold}#0", f"sNa{i}#0"),
            selected_turn_ids=(f"{gold}#0",), truncated=False,
            prompt_parts=(
                ("instruction", "Answer the question based on the context."),
                ("context", f"turn {gold}#0"),
            ),
            reader_correct_when_contains=RECONCILE_MARKER,
        ))
    return insts


def _build_chaotic_author():
    """A ChaoticMockAuthor scripted to reproduce the real failure mix:
    budget-truncation gets a discard / sandbox-reject / static-reject mix
    (exhausts → rotate), then assemble-internal gets a promotable
    reader-prompt-transform."""
    from regimes.targets.longmemeval.mock_author import ChaoticMockAuthor

    return ChaoticMockAuthor(by_regime={
        "budget-truncation": ["discard", "sandbox_reject", "static_reject"],
        "assemble-internal": ["promote"],
    })


def _build_mock_instances() -> list[MockInstance]:
    """A small fixture mix used to exercise the histogram in --mock
    mode. Carries one of each main regime plus a couple of correct
    answers so per-type accuracy is non-trivial."""
    return [
        MockInstance("q_ok1", "multi-session", False, ("s1",), True,
                     scores={"s1#0": 1.0}, selected_turn_ids=("s1#0",)),
        MockInstance("q_ok2", "single-session-user", False, ("s2",), True,
                     scores={"s2#0": 0.9}, selected_turn_ids=("s2#0",)),
        MockInstance("q_ok3", "knowledge-update", False, ("s3",), True,
                     scores={"s3#0": 0.95}, selected_turn_ids=("s3#0",)),
        MockInstance(
            "q_se", "multi-session", False, ("sX",), False,
            scores={},
            score_error="agent.score_embedding:BadRequestError: input too long",
        ),
        MockInstance(
            "q_ac1", "multi-session", False, ("sG_ac",), False,
            scores={"sG_ac#0": 0.6, "sN#0": 0.95},
            ranked=("sN#0", "sG_ac#0"),
            selected_turn_ids=("sN#0",), truncated=True,
            # Annotated so stub_topk_boost flips this instance to correct
            # (sG_ac#0 0.6 * 1.25 = 0.75 >= threshold). Makes mock --full
            # produce a real promotion, which is what exercises the CONFIRM
            # held-out gate end-to-end.
            gold_score_threshold=0.7,
            candidate_turn_ids=("sG_ac#0",),
        ),
        MockInstance(
            "q_bt1", "temporal-reasoning", False, ("sG_bt",), False,
            scores={"sG_bt#0": 0.8, "sM#0": 0.9},
            ranked=("sM#0", "sG_bt#0"),
            selected_turn_ids=("sM#0",), truncated=True,
            decisions=(
                {"turn_id": "sG_bt#0", "included": False, "reason": "budget"},
            ),
        ),
        MockInstance(
            "q_sg", "temporal-reasoning", False, ("sG_sg",), False,
            scores={"sG_sg#0": 0.01,
                    **{f"oN{j}#0": 0.5 for j in range(25)}},
            ranked=tuple(f"oN{j}#0" for j in range(25)) + ("sG_sg#0",),
            selected_turn_ids=tuple(f"oN{j}#0" for j in range(5)),
        ),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("mock", "real"), default="mock")
    p.add_argument(
        "--full", action="store_true",
        help="Run all phases through stop (default: pause after histogram).",
    )
    p.add_argument(
        "--chaotic", action="store_true",
        help="Mock mode only: use the ChaoticMockAuthor (emits the real "
             "author's failure mix) against a budget-truncation + "
             "assemble-internal fixture so the loop rotates between two "
             "seam-reachable regimes and ends cleanly.",
    )
    p.add_argument("--run-dir", default="runs/loop_001",
                   help="Where to write the report.")
    p.add_argument("--split", default="config/split.json")
    p.add_argument(
        "--split-seed", type=int, default=None,
        help="Real mode only: resample a fresh OPTIMIZE/CONFIRM split with "
             "this seed (same stratification rule, a DIFFERENT held-out draw) "
             "instead of loading --split. Writes config/split.seed<seed>.json "
             "and uses it; the committed config/split.json is left untouched.",
    )
    p.add_argument("--lme-checkout", default=str(REPO.parent / "activegraph-longmemeval"))
    p.add_argument("--lme-data", default=None,
                   help="Path to longmemeval_s_cleaned.json (real mode only).")
    p.add_argument("--signal", default="embedding")
    p.add_argument("--token-budget", type=int, default=2500)
    args = p.parse_args()

    pause_after = None if args.full else "histogram"
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "mock":
        backend = MockEval()
        if args.chaotic:
            instances = _build_chaotic_mock_instances()
            confirm = _build_mock_confirm_instances()
            rep = run_loop(eval_backend=backend, instances=instances,
                           confirm_instances=confirm,
                           author=_build_chaotic_author(),
                           pause_after=pause_after)
        else:
            instances = _build_mock_instances()
            confirm = _build_mock_confirm_instances()
            rep = run_loop(eval_backend=backend, instances=instances,
                           confirm_instances=confirm,
                           pause_after=pause_after)
    else:
        # Real path: build a RealEval, feed it the OPTIMIZE instances
        # for hypothesize/eval-diff and the CONFIRM instances for the
        # promotion gate's held-out check.
        if not args.lme_data:
            sys.stderr.write(
                "--lme-data is required in --mode real "
                "(point at longmemeval_s_cleaned.json)\n"
            )
            return 2
        from regimes.agent import OpenAIEmbedder, set_embedder  # noqa: WPS433
        from regimes.eval import LMEJudge, RealEval, build_real_reader  # noqa: WPS433
        set_embedder(OpenAIEmbedder())
        if args.split_seed is not None:
            # Resample a fresh held-out draw from the real dataset with a
            # different seed. Reuses the canonical builder so stratification
            # is identical; writes to a seeded path so config/split.json is
            # never overwritten.
            sys.path.insert(0, str(REPO / "scripts"))
            from build_split import build_split as _build_split_file  # noqa: WPS433
            split_path = Path(f"config/split.seed{args.split_seed}.json")
            _build_split_file(Path(args.lme_data), split_path, seed=args.split_seed)
            s = load_split(split_path)
        else:
            s = load_split(args.split)
        by_id = {x["question_id"]: x for x in json.load(open(args.lme_data))}
        opt = [by_id[q] for q in s.optimize]
        confirm = [by_id[q] for q in s.confirm]
        backend = RealEval(
            reader=build_real_reader(),
            judge=LMEJudge(lme_checkout=args.lme_checkout),
            signal=args.signal, token_budget=args.token_budget,
        )
        # RealEval needs run_dir on its run_on_split. Wrap it.
        class _RD:
            def __init__(self, ev, base): self.ev, self.base, self.n = ev, base, 0
            def run_on_split(self, insts):
                self.n += 1
                return self.ev.run_on_split(insts, run_dir=self.base / f"sub_{self.n}")
        wrapped = _RD(backend, run_dir)
        # Real mode authors transforms with Claude. `BEHAVIORDRAFTS_MODEL`
        # overrides the model id; ANTHROPIC_API_KEY must be present (the
        # constructor raises ConfigurationError otherwise — caller-fixable).
        author = build_real_author()
        rep = run_loop(eval_backend=wrapped, instances=opt,
                       confirm_instances=confirm,
                       author=author, pause_after=pause_after)

    # --- print + persist ---
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
            print(f"  {r['status']:18s} {r['name']:24s} "
                  f"target={r['target_regime']:20s} "
                  f"reasons={r.get('reasons', '')}")
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
    }
    out_path = run_dir / "report.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"report → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
