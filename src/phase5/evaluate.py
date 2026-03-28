"""
Phase 5 — Evaluation Script
Computes BLEU-4 and ROUGE-L scores to measure report quality.
Also runs the ablation: compares 1 / 3 / 5 keyframe outputs.

Usage:
    # Score generated reports against reference descriptions
    python3 src/phase5/evaluate.py --refs data/references/ --gens outputs/reports/

    # Run ablation on a single video
    python3 src/phase5/evaluate.py --ablation --video data/samples/test.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ─── Ensure NLTK punkt downloaded ─────────────────────────────────────────────

def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)


# ─── Tokenizer ────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    try:
        return nltk.word_tokenize(text.lower())
    except Exception:
        return text.lower().split()


# ─── BLEU-4 ──────────────────────────────────────────────────────────────────

def compute_bleu(references: list[str], hypotheses: list[str]) -> float:
    """
    Corpus BLEU-4 between reference and hypothesis sentence lists.
    Returns float [0, 1].
    """
    refs_tok = [[tokenize(r)] for r in references]
    hyps_tok = [tokenize(h) for h in hypotheses]
    smooth = SmoothingFunction().method1
    score = corpus_bleu(refs_tok, hyps_tok, smoothing_function=smooth)
    return round(float(score), 4)


# ─── ROUGE-L ─────────────────────────────────────────────────────────────────

def compute_rouge_l(references: list[str], hypotheses: list[str]) -> float:
    """
    Average ROUGE-L F1 between reference and hypothesis lists.
    Returns float [0, 1].
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for ref, hyp in zip(references, hypotheses):
        result = scorer.score(ref, hyp)
        scores.append(result["rougeL"].fmeasure)
    return round(sum(scores) / len(scores) if scores else 0.0, 4)


# ─── Load reports from directory ─────────────────────────────────────────────

def load_prose_reports(report_dir: str) -> list[str]:
    """Load prose reports from JSON files in a directory."""
    reports = []
    for fp in sorted(Path(report_dir).glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
        prose = data.get("prose_report", "")
        if prose:
            reports.append(prose)
    return reports


def load_reference_texts(ref_dir: str) -> list[str]:
    """Load reference descriptions from .txt files."""
    texts = []
    for fp in sorted(Path(ref_dir).glob("*.txt")):
        texts.append(fp.read_text().strip())
    return texts


# ─── Ablation Study ──────────────────────────────────────────────────────────

def run_ablation(video_path: str, vlm_model: str = "minicpm-v", llm_model: str = "mistral"):
    """
    Run the full pipeline with 1, 3, and 5 keyframes and report metrics.
    Uses VLM narrative length + inference time as proxy metrics when
    no ground truth is available.
    """
    from src.phase3.keyframe_sampler import sample_keyframes
    from src.phase3.vlm_narrator import VLMNarrator
    from src.phase3.report_generator import ReportGenerator

    narrator = VLMNarrator(model=vlm_model)
    reporter = ReportGenerator(model=llm_model)

    results = []
    print("\n" + "=" * 60)
    print("  ABLATION STUDY: Keyframe Count Comparison")
    print("=" * 60)
    print(f"  Video : {video_path}")
    print(f"  VLM   : {vlm_model}  |  LLM: {llm_model}")
    print("-" * 60)

    header = f"{'N':>4} | {'Infer(s)':>9} | {'Severity':>10} | {'Flag':>6} | {'Prose words':>12}"
    print(header)
    print("-" * 60)

    for n in [1, 3, 5]:
        kf_dir = f"/tmp/ablation_kf_{n}"
        keyframes = sample_keyframes(video_path, n=n, output_dir=kf_dir)
        if not keyframes:
            print(f"  {n:>4} | {'ERROR':>9}")
            continue

        frames = [f for _, f in keyframes]
        t0 = time.time()
        inc_json = narrator.narrate(frames)
        t1 = time.time()

        prose = reporter.generate(inc_json)
        word_count = len(prose.split())
        severity = inc_json.get("severity", "?")
        flag = "⚠ Yes" if inc_json.get("confidence_flag") else "✓ No"
        infer_s = round(t1 - t0, 1)

        results.append({
            "n_keyframes": n,
            "inference_time_s": infer_s,
            "severity": severity,
            "confidence_flag": inc_json.get("confidence_flag"),
            "prose_word_count": word_count,
            "prose": prose,
        })
        print(f"  {n:>4} | {infer_s:>9} | {severity:>10} | {flag:>6} | {word_count:>12}")

    print("=" * 60)
    print("\nConclusion:")
    if results:
        fastest = min(results, key=lambda x: x["inference_time_s"])
        most_words = max(results, key=lambda x: x["prose_word_count"])
        print(f"  Fastest inference : {fastest['n_keyframes']} keyframe(s) ({fastest['inference_time_s']}s)")
        print(f"  Most detailed     : {most_words['n_keyframes']} keyframe(s) ({most_words['prose_word_count']} words)")
    print()
    return results


# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_reports(ref_dir: str, gen_dir: str):
    _ensure_nltk()
    refs = load_reference_texts(ref_dir)
    hyps = load_prose_reports(gen_dir)

    if not refs or not hyps:
        print("[Eval] No references or generated reports found.")
        return

    n = min(len(refs), len(hyps))
    refs, hyps = refs[:n], hyps[:n]

    bleu = compute_bleu(refs, hyps)
    rouge = compute_rouge_l(refs, hyps)

    print("\n" + "=" * 55)
    print("  Evaluation Results")
    print("=" * 55)
    print(f"  Pairs scored : {n}")
    print(f"  BLEU-4       : {bleu:.4f}  ({bleu*100:.1f}%)")
    print(f"  ROUGE-L F1   : {rouge:.4f}  ({rouge*100:.1f}%)")
    print("=" * 55)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Evaluation & Ablation")
    subparsers = parser.add_subparsers(dest="command")

    # score command
    score_p = subparsers.add_parser("score", help="Score generated reports vs. references")
    score_p.add_argument("--refs", required=True, help="Directory of reference .txt files")
    score_p.add_argument("--gens", required=True, help="Directory of generated report .json files")

    # ablation command
    abl_p = subparsers.add_parser("ablation", help="Ablation study: 1/3/5 keyframes")
    abl_p.add_argument("--video", required=True, help="Dashcam video path")
    abl_p.add_argument("--vlm", default="minicpm-v")
    abl_p.add_argument("--llm", default="mistral")

    args = parser.parse_args()

    if args.command == "score":
        score_reports(args.refs, args.gens)
    elif args.command == "ablation":
        run_ablation(args.video, args.vlm, args.llm)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
