"""Benchmark Runner: Evaluates Baselines vs Proposed Pipeline and generates Deliverables in < 15 mins."""

import json
import time
from typing import List, Dict, Any
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.models import TweetInput
from src.config import REPORT_OUTPUT_PATH, BENCHMARK_SUMMARY_JSON_PATH, TARGET_BRAND
from src.eval.splits import CALIBRATION, HELDOUT, load_golden_rows, split_counts
from src.pipeline import SupportPipeline
from src.intent.baselines import TrivialMajorityClassifier, SimpleTfidfClassifier
from src.eval.metrics import compute_intent_metrics, compute_triage_metrics, compute_rouge_similarity
from src.eval.judge import LLMJudge
from src.eval.human_agreement import compute_human_judge_agreement
from src.eval.report_generator import generate_markdown_report, write_benchmark_summary_json, _abbreviate_labels
from src.eval.failure_analysis import mine_failure_modes

app = typer.Typer(help="Hiver AI Support Agent Evaluation Harness")
console = Console()


def load_golden_dataset(limit: int = None, split: str = None) -> List[Dict]:
    """Loads the hand-labelled golden dataset from JSONL. `limit` truncates
    to a subset (used by --quick); the full file is used otherwise --
    previously this hardcoded 200 regardless of how many rows the golden
    set actually contained, silently dropping rows if the file grew past
    200 or padding metrics with nothing if it shrank below 200.

    `split` restricts to "calibration" or "heldout" (src/eval/splits.py).
    Headline numbers are reported on the held-out rows only, because the
    thresholds in src/config.py were tuned against the calibration rows --
    scoring on the data a threshold was chosen from is the bias this split
    exists to remove."""
    return load_golden_rows(split=split, limit=limit)


@app.command()
def run(
    quick: bool = typer.Option(False, "--quick", "-q", help="Run on smaller 50-sample subset for ultra-fast verification"),
):
    """Executes the complete benchmark evaluation comparing Proposed System against 2 Baselines."""
    start_total = time.perf_counter()
    counts = split_counts()
    heldout_rows = load_golden_dataset(split=HELDOUT)
    calibration_rows = load_golden_dataset(split=CALIBRATION)
    total_n = counts.get(HELDOUT, 0) + counts.get(CALIBRATION, 0)
    limit = 50 if quick else len(heldout_rows)

    console.print(Panel(
        f"[bold cyan]Hiver SDE Intern Benchmark Evaluation Runner[/bold cyan]\n"
        f"Target Brand: [bold white]{TARGET_BRAND}[/bold white]\n"
        f"Headline Set: [bold yellow]{min(limit, len(heldout_rows))} of {counts.get(HELDOUT, 0)} HELD-OUT rows[/bold yellow]\n"
        f"Calibration Set (thresholds tuned here, reported separately): "
        f"[bold yellow]{counts.get(CALIBRATION, 0)} rows[/bold yellow]\n"
        f"Golden set total: [bold white]{total_n}[/bold white]\n"
        f"Requirement: [bold green]Reproducible in < 15 minutes[/bold green]",
        title="[bold green]Benchmark Suite[/bold green]",
        expand=False
    ))

    data = heldout_rows[:limit]
    y_true_intent = [d["true_intent"] for d in data]
    y_true_triage = [d["true_triage_action"] for d in data]
    references = [d["reference_resolution"] for d in data]

    # -------------------------------------------------------------
    # 1. Evaluate Baseline 1: Trivial (Majority Class + Canned Reply)
    # -------------------------------------------------------------
    console.print("[dim]Evaluating Baseline 1 (Trivial Majority)...[/dim]")
    b1_clf = TrivialMajorityClassifier()
    b1_preds_intent = [b1_clf.predict(d["text"]).primary_intent.value for d in data]
    b1_preds_triage = ["AUTO_HANDLE" for _ in data]  # Naively assumes everything can be auto-handled
    b1_replies = ["Thanks for reaching out. We'd like to help. Please restart your device." for _ in data]

    b1_intent_metrics = compute_intent_metrics(y_true_intent, b1_preds_intent)
    b1_triage_metrics = compute_triage_metrics(y_true_triage, b1_preds_triage)
    b1_rouge = compute_rouge_similarity(references, b1_replies)
    trivial_results = {"intent": b1_intent_metrics, "triage": b1_triage_metrics, "rouge": b1_rouge}

    # -------------------------------------------------------------
    # 2. Evaluate Baseline 2: Simple (TF-IDF + Simple Retrieval)
    # -------------------------------------------------------------
    console.print("[dim]Evaluating Baseline 2 (Simple TF-IDF)...[/dim]")
    b2_clf = SimpleTfidfClassifier()
    b2_preds_intent = [b2_clf.predict(d["text"]).primary_intent.value for d in data]
    b2_preds_triage = []
    for d in data:
        # Simple heuristic: escalate if keywords appear
        t = d["text"].lower()
        if any(w in t for w in ["help", "broken", "human", "agent", "lawyer", "refund"]):
            b2_preds_triage.append("ESCALATE")
        else:
            b2_preds_triage.append("AUTO_HANDLE")
    b2_replies = [f"We can help with your Apple issue. Please check apple.co for troubleshooting." for _ in data]

    b2_intent_metrics = compute_intent_metrics(y_true_intent, b2_preds_intent)
    b2_triage_metrics = compute_triage_metrics(y_true_triage, b2_preds_triage)
    b2_rouge = compute_rouge_similarity(references, b2_replies)
    simple_results = {"intent": b2_intent_metrics, "triage": b2_triage_metrics, "rouge": b2_rouge}

    # -------------------------------------------------------------
    # 3. Evaluate Proposed System (Production Pipeline)
    # -------------------------------------------------------------
    console.print("[dim]Evaluating Proposed Production Pipeline (Semantic RAG + Triage Gate)...[/dim]")
    pipeline = SupportPipeline()

    tweets = [TweetInput(tweet_id=d["tweet_id"], text=d["text"], author_id=d["author_id"]) for d in data]
    prod_responses = pipeline.batch_process(tweets)

    prod_preds_intent = [r.intent.primary_intent.value for r in prod_responses]
    prod_preds_triage = [r.triage.action.value for r in prod_responses]
    prod_replies = [r.drafted_reply or "" for r in prod_responses]

    prod_intent_metrics = compute_intent_metrics(y_true_intent, prod_preds_intent)
    prod_triage_metrics = compute_triage_metrics(y_true_triage, prod_preds_triage)
    prod_rouge = compute_rouge_similarity(references, prod_replies)
    prod_results = {"intent": prod_intent_metrics, "triage": prod_triage_metrics, "rouge": prod_rouge}

    # -------------------------------------------------------------
    # 3b. Same pipeline, CALIBRATION rows -- printed beside the held-out
    #     numbers so the gap between "data the thresholds were tuned on" and
    #     "data they were not" is visible instead of being a footnote nobody
    #     can check. A large gap here means the held-out numbers are the only
    #     ones worth quoting.
    # -------------------------------------------------------------
    console.print("[dim]Evaluating the same pipeline on the calibration split (for the generalisation gap)...[/dim]")
    cal_tweets = [TweetInput(tweet_id=d["tweet_id"], text=d["text"], author_id=d["author_id"]) for d in calibration_rows]
    cal_responses = pipeline.batch_process(cal_tweets)
    cal_intent_metrics = compute_intent_metrics(
        [d["true_intent"] for d in calibration_rows],
        [r.intent.primary_intent.value for r in cal_responses],
    )
    cal_triage_metrics = compute_triage_metrics(
        [d["true_triage_action"] for d in calibration_rows],
        [r.triage.action.value for r in cal_responses],
    )
    split_info = {
        "heldout_n": len(data),
        "calibration_n": len(calibration_rows),
        "seed": __import__("src.eval.splits", fromlist=["SPLIT_SEED"]).SPLIT_SEED,
        "calibration_intent_accuracy": cal_intent_metrics["accuracy"],
        "calibration_triage_accuracy": cal_triage_metrics["accuracy"],
        "calibration_escalation_recall": cal_triage_metrics["escalation_recall"],
        "heldout_label_counts": {
            lbl: sum(1 for d in data if d["true_triage_action"] == lbl)
            for lbl in sorted({d["true_triage_action"] for d in heldout_rows + calibration_rows})
        },
    }

    # -------------------------------------------------------------
    # 4. LLM-as-a-Judge Evaluation & Human Calibration
    # -------------------------------------------------------------
    console.print("[dim]Evaluating LLM-as-a-Judge Rubric & Human Agreement...[/dim]")
    judge = LLMJudge()
    judge_sample_size = min(50, len(data))

    def _judge_avg(replies_subset, texts_subset, refs_subset):
        scores = [
            judge.grade_reply(t, rep, ref)["overall"]
            for t, rep, ref in zip(texts_subset, replies_subset, refs_subset)
        ]
        # No fabricated fallback constant: an empty sample means the score
        # is genuinely undefined, not "4.5" or any other plausible-looking
        # number, so report 0.0 with the sample size making that visible.
        return round(sum(scores) / len(scores), 2) if scores else 0.0

    judge_texts = [d["text"] for d in data[:judge_sample_size]]
    judge_refs = [d["reference_resolution"] for d in data[:judge_sample_size]]

    # Previously only the production system's replies were ever graded by
    # the judge -- the "2.1 / 5.0" and "3.4 / 5.0" baseline scores shown in
    # the report and terminal table were typed-in literals, not measured.
    # Grade all three systems' replies on the same sample so the comparison
    # is real.
    b1_judge_results = {"overall_score": _judge_avg(b1_replies[:judge_sample_size], judge_texts, judge_refs)}
    b2_judge_results = {"overall_score": _judge_avg(b2_replies[:judge_sample_size], judge_texts, judge_refs)}
    judge_results = {"overall_score": _judge_avg(prod_replies[:judge_sample_size], judge_texts, judge_refs)}

    trivial_results["judge"] = b1_judge_results
    simple_results["judge"] = b2_judge_results
    prod_results["judge"] = judge_results

    agreement_results = compute_human_judge_agreement()

    # -------------------------------------------------------------
    # 5. Extract Top 5 Failure Modes
    # -------------------------------------------------------------
    failures = []
    for d, r in zip(data, prod_responses):
        if r.intent.primary_intent.value != d["true_intent"] or r.triage.action.value != d["true_triage_action"]:
            failures.append({
                "tweet_id": d["tweet_id"],
                "text": d["text"],
                "true_intent": d["true_intent"],
                "pred_intent": r.intent.primary_intent.value,
                "true_triage": d["true_triage_action"],
                "pred_triage": r.triage.action.value,
                "stated_reason": r.triage.stated_reason,
            })

    # Previously this was a hardcoded list of 5 plausible-sounding failure
    # modes (specific frequencies, specific example queries) that had no
    # connection to the `failures` list computed just above -- that list was
    # computed and then never read. Now the real failures from this actual
    # run are clustered and reported; if the pipeline made zero mistakes on
    # this run, this returns an empty list (an empty section in the report
    # is the honest outcome, not a reason to fall back to invented examples).
    top_5_failure_analysis = mine_failure_modes(failures, top_n=5)

    # Real measured P95 latency across this run's production responses --
    # not the "< 35 ms" narrative figure that used to be hand-typed into
    # both docs/REPORT.md and the dashboard. Computed here (not inside
    # report_generator.py) since it needs the raw per-response timings,
    # which only this function has. Computed before both the markdown
    # report and the JSON summary so neither one can silently disagree
    # with the other about what was actually measured.
    prod_latencies_ms = [r.execution_time_ms for r in prod_responses if r.execution_time_ms is not None]
    latency_p95_ms = (
        round(float(np.percentile(prod_latencies_ms, 95)), 2) if prod_latencies_ms else None
    )

    # -------------------------------------------------------------
    # 6. Generate Formal Report docs/REPORT.md
    # -------------------------------------------------------------
    generate_markdown_report(
        trivial_metrics=trivial_results,
        simple_metrics=simple_results,
        prod_metrics=prod_results,
        judge_metrics=judge_results,
        agreement_metrics=agreement_results,
        top_failures=top_5_failure_analysis,
        latency_p95_ms=latency_p95_ms,
        split_info=split_info,
    )

    benchmark_summary = write_benchmark_summary_json(
        trivial_metrics=trivial_results,
        simple_metrics=simple_results,
        prod_metrics=prod_results,
        agreement_metrics=agreement_results,
        latency_p95_ms=latency_p95_ms,
        split_info=split_info,
    )

    elapsed_total = time.perf_counter() - start_total

    # -------------------------------------------------------------
    # 7. Print Terminal Headline Comparison Table
    # -------------------------------------------------------------
    table = Table(title="[bold green]Hiver SDE Intern: Headline Benchmark Results[/bold green]", show_header=True)
    table.add_column("Evaluation Metric", style="cyan", no_wrap=True)
    table.add_column("Baseline 1 (Trivial)", justify="center")
    table.add_column("Baseline 2 (Simple)", justify="center")
    table.add_column("Proposed System (Production)", justify="center", style="bold green")
    table.add_column("Lift vs Simple", justify="center", style="bold yellow")

    # NOTE: every "+{...:.4f}" literal below was a sign bug, not just a
    # style choice -- when the production system scores *worse* than the
    # Simple baseline on a given metric (which can legitimately happen,
    # especially on a small/rebuilt golden set), the hardcoded "+" prefix
    # printed a negative number as "+-0.0123", misreporting a regression as
    # a gain. The ":+" format spec below prints the sign the value actually
    # has.
    table.add_row(
        "Intent Macro-F1",
        f"{b1_intent_metrics['macro_f1']:.4f}",
        f"{b2_intent_metrics['macro_f1']:.4f}",
        f"{prod_intent_metrics['macro_f1']:.4f}",
        f"{(prod_intent_metrics['macro_f1'] - b2_intent_metrics['macro_f1']):+.4f}",
    )
    table.add_row(
        "Intent Accuracy",
        f"{b1_intent_metrics['accuracy']*100:.1f}%",
        f"{b2_intent_metrics['accuracy']*100:.1f}%",
        f"{prod_intent_metrics['accuracy']*100:.1f}%",
        f"{(prod_intent_metrics['accuracy'] - b2_intent_metrics['accuracy'])*100:+.1f}%",
    )
    table.add_row(
        "Triage Accuracy",
        f"{b1_triage_metrics['accuracy']*100:.1f}%",
        f"{b2_triage_metrics['accuracy']*100:.1f}%",
        f"{prod_triage_metrics['accuracy']*100:.1f}%",
        f"{(prod_triage_metrics['accuracy'] - b2_triage_metrics['accuracy'])*100:+.1f}%",
    )
    table.add_row(
        "Escalation Recall",
        f"{b1_triage_metrics['escalation_recall']*100:.1f}%",
        f"{b2_triage_metrics['escalation_recall']*100:.1f}%",
        f"{prod_triage_metrics['escalation_recall']*100:.1f}%",
        f"{(prod_triage_metrics['escalation_recall'] - b2_triage_metrics['escalation_recall'])*100:+.1f}%",
    )
    # Denominator was a hardcoded literal "/ 30" regardless of how many true
    # ESCALATE rows the loaded golden set actually contained. All three
    # systems are scored against the same y_true_triage, so the true count
    # is identical across b1/b2/prod -- pull it once from the real metrics.
    total_esc_true = prod_triage_metrics["total_escalations_true"]
    table.add_row(
        "Missed Escalations (Safety Risk)",
        f"{b1_triage_metrics['missed_escalation_count']} / {total_esc_true}",
        f"{b2_triage_metrics['missed_escalation_count']} / {total_esc_true}",
        f"[bold green]{prod_triage_metrics['missed_escalation_count']} / {total_esc_true}[/bold green]",
        f"{(b2_triage_metrics['missed_escalation_count'] - prod_triage_metrics['missed_escalation_count']):+d}",
    )
    rouge_lift = prod_rouge['mean_rougeL'] - b2_rouge['mean_rougeL']
    table.add_row(
        "ROUGE-L Grounding Score",
        f"{b1_rouge['mean_rougeL']:.4f}",
        f"{b2_rouge['mean_rougeL']:.4f}",
        f"{prod_rouge['mean_rougeL']:.4f}",
        f"{rouge_lift:+.4f}",
    )
    judge_lift = judge_results['overall_score'] - b2_judge_results['overall_score']
    table.add_row(
        "LLM Judge Quality (1-5)",
        f"{b1_judge_results['overall_score']:.1f} / 5.0",
        f"{b2_judge_results['overall_score']:.1f} / 5.0",
        f"{judge_results['overall_score']:.1f} / 5.0",
        f"{judge_lift:+.1f}",
    )

    console.print(table)

    # -------------------------------------------------------------
    # 7a. Generalisation gap: tuned-on vs reported-on
    # -------------------------------------------------------------
    gap_table = Table(
        title="[bold yellow]Generalisation gap: calibration split (thresholds tuned here) vs held-out (reported above)[/bold yellow]",
        show_header=True,
    )
    gap_table.add_column("Metric", style="cyan", no_wrap=True)
    gap_table.add_column(f"Calibration (n={len(calibration_rows)})", justify="center")
    gap_table.add_column(f"Held-out (n={len(data)})", justify="center", style="bold green")
    gap_table.add_column("Gap", justify="center", style="bold yellow")
    for label, cal_v, held_v in (
        ("Intent Accuracy", cal_intent_metrics["accuracy"], prod_intent_metrics["accuracy"]),
        ("Triage Accuracy", cal_triage_metrics["accuracy"], prod_triage_metrics["accuracy"]),
        ("Escalation Recall", cal_triage_metrics["escalation_recall"], prod_triage_metrics["escalation_recall"]),
    ):
        gap_table.add_row(label, f"{cal_v*100:.1f}%", f"{held_v*100:.1f}%", f"{(held_v - cal_v)*100:+.1f}%")
    console.print(gap_table)
    console.print(
        "[dim]A held-out number below its calibration counterpart is the expected direction: "
        "thresholds were chosen against the calibration rows. Quote the held-out column.[/dim]"
    )

    # -------------------------------------------------------------
    # 7b. Print the Production Intent Confusion Matrix
    # -------------------------------------------------------------
    # compute_intent_metrics() has always computed this (see
    # src/eval/metrics.py) but nothing ever printed or reported it -- the
    # same "computed, never read" pattern this codebase's audit already
    # found once with the per-example `failures` list below. It's also
    # written into docs/REPORT.md Section 4.1 and into
    # docs/benchmark_summary.json for the dashboard; this terminal table is
    # the same numbers, for immediate visibility without opening either file.
    cm_labels = prod_intent_metrics.get("labels", [])
    cm_matrix = prod_intent_metrics.get("confusion_matrix", [])
    if cm_labels and cm_matrix:
        abbrevs = _abbreviate_labels(cm_labels)
        cm_table = Table(title="[bold magenta]Production Intent Confusion Matrix (rows=true, cols=predicted)[/bold magenta]", show_header=True)
        cm_table.add_column("True \\ Pred", style="cyan", no_wrap=True)
        for lbl in cm_labels:
            cm_table.add_column(abbrevs[lbl], justify="center")
        for i, true_label in enumerate(cm_labels):
            row_cells = []
            for j, _ in enumerate(cm_labels):
                count = cm_matrix[i][j]
                row_cells.append(f"[bold green]{count}[/bold green]" if i == j else (str(count) if count else "[dim]-[/dim]"))
            cm_table.add_row(f"{abbrevs[true_label]}", *row_cells)
        console.print(cm_table)
        console.print("[dim]" + "  ".join(f"{v}={k}" for k, v in abbrevs.items()) + "[/dim]")

    # Print Calibration & Summary
    summary_panel = Panel(
        f"[bold]Human-Judge Cohen's Kappa:[/bold] [bold green]kappa = {agreement_results['mean_cohen_kappa']:.4f}[/bold green] ({agreement_results['agreement_interpretation']})\n"
        f"[bold]Exact Agreement (Safety):[/bold] [bold green]{agreement_results['exact_agreement_safety_pct']:.1f}%[/bold green]\n"
        f"[bold]Report Generated:[/bold] [underline cyan]{REPORT_OUTPUT_PATH}[/underline cyan]\n"
        f"[bold]Dashboard Summary JSON:[/bold] [underline cyan]{BENCHMARK_SUMMARY_JSON_PATH}[/underline cyan]\n"
        f"[bold]Production P95 Latency (measured):[/bold] [bold yellow]{latency_p95_ms} ms[/bold yellow]\n"
        f"[bold]Benchmark Execution Time:[/bold] [bold yellow]{elapsed_total:.2f} seconds[/bold yellow] (Target: < 900s / 15 mins)\n",
        title="[bold green]Verification & Deliverables Summary[/bold green]",
        expand=False,
    )
    console.print(summary_panel)


if __name__ == "__main__":
    app()
