"""
run_workflow.py -- Unified Workflow CLI for Neurosymbolic Clinical Extraction.

Provides a single consolidated entrypoint to execute:
  1. Single document closed-loop extraction and verification.
  2. Dataset A (400-document synthetic ablation benchmark).
  3. Dataset B (180-document realistic multi-template cohort).
  4. Publication/presentation figures generation.
  5. Review 2 slide deck population.
  6. Full end-to-end milestone workflow.

Usage:
  python run_workflow.py --mode single --image data/ground_truth/sample_002.png
  python run_workflow.py --mode eval-a
  python run_workflow.py --mode eval-b
  python run_workflow.py --mode plots
  python run_workflow.py --mode slides
  python run_workflow.py --mode full
"""

import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import run_pipeline


def execute_single(image_path: str):
    print("\n" + "=" * 74)
    print("  SINGLE DOCUMENT EXTRACTION & SELF-CORRECTION WORKFLOW")
    print("=" * 74)
    if not os.path.isfile(image_path):
        print(f"Error: Target image file not found: {image_path}")
        sys.exit(1)

    print(f"  Target Image : {image_path}")
    result = run_pipeline(image_path)

    status = result.get("status")
    route = result.get("assay_route")
    attempts = result.get("attempts", 0)
    data = result.get("data", {})
    audit = result.get("audit_trail", [])

    print(f"  Final Status : {status}")
    print(f"  Assay Route  : {route}")
    print(f"  Iterations   : {attempts}")
    if "residual" in result:
        print(f"  Residual     : {result['residual']} mg/dL")
    if "relative_error" in result:
        print(f"  Rel Error    : {result['relative_error']:.2%}")

    print("\n  Canonical Extracted Values (mg/dL):")
    print("  " + "-" * 40)
    for k, v in data.items():
        print(f"    {k:<22} : {v:>7.1f} mg/dL")

    print("\n  Audit Trail Steps:")
    print("  " + "-" * 70)
    if not audit:
        print("    [Clean Invariant Pass - Zero Mutations Required]")
    else:
        for idx, step in enumerate(audit, 1):
            stg = step.get("stage", "STAGE_GENERAL")
            if "field" in step:
                f = step["field"]
                old = step.get("old_val")
                new = step.get("new_val")
                d_c = step.get("delta_conf", "N/A")
                print(f"    Step {idx}: [{stg}] Field: {f:<18} | {old} -> {new} (Delta-conf: {d_c})")
            elif "modifications" in step:
                mods = step["modifications"]
                mod_str = ", ".join([f"{m['field']}: {m['old_val']}->{m['new_val']}" for m in mods])
                print(f"    Step {idx}: [{stg}] Pair: {mod_str}")
            else:
                print(f"    Step {idx}: [{stg}] Action: {step.get('action', step.get('reason', ''))}")
    print("=" * 74 + "\n")


def execute_eval_a(verbose: bool = False):
    from evals.run_ablations import run_eval, print_summary
    results = run_eval(verbose=verbose)
    print_summary(results)


def execute_eval_b(verbose: bool = False):
    from evals.eval_dataset_b import run_eval_dataset_b, print_summary_dataset_b
    results = run_eval_dataset_b(verbose=verbose)
    print_summary_dataset_b(results)


def execute_plots():
    from evals.generate_plots import generate_all_plots
    generate_all_plots()


def execute_slides():
    print("=" * 70)
    print("  GENERATING REVIEW 2 PRESENTATION DECK")
    print("=" * 70)
    cmd = [sys.executable, os.path.join(ROOT, "generate_review2_slides.py")]
    subprocess.run(cmd, check=True)
    print("=" * 70 + "\n")


def execute_full():
    print("\n" + "#" * 74)
    print("  EXECUTING COMPLETE CLINICAL EXTRACTION PROJECT WORKFLOW")
    print("#" * 74 + "\n")

    print(">>> STAGE 1: Evaluating Dataset A (400-doc Synthetic Benchmark)")
    execute_eval_a(verbose=False)

    print("\n>>> STAGE 2: Evaluating Dataset B (180-doc Realistic Multi-Template Cohort)")
    execute_eval_b(verbose=False)

    print("\n>>> STAGE 3: Generating Publication Figures")
    execute_plots()

    print("\n>>> STAGE 4: Generating Review 2 Populated Slide Deck")
    execute_slides()

    print("\n" + "#" * 74)
    print("  ALL WORKFLOW PHASES COMPLETED SUCCESSFULLY")
    print("#" * 74 + "\n")


def execute_reproduce():
    from src.stage5_reproduce import run_stage5_reproduction
    run_stage5_reproduction()


def main():
    parser = argparse.ArgumentParser(
        description="Clinical Extraction Loop -- Unified Project Workflow"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "eval-a", "eval-b", "plots", "slides", "full", "reproduce"],
        default="full",
        help="Workflow execution mode"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to clinical document image for --mode single"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed per-document console output during evaluation"
    )

    args = parser.parse_args()

    if args.mode == "single":
        if not args.image:
            # Default to a representative test image
            default_img = os.path.join(ROOT, "data", "ground_truth", "sample_002.png")
            if os.path.isfile(default_img):
                print(f"No --image specified, defaulting to {default_img}")
                args.image = default_img
            else:
                parser.error("--image is required when --mode single is selected")
        execute_single(args.image)
    elif args.mode == "eval-a":
        execute_eval_a(verbose=args.verbose)
    elif args.mode == "eval-b":
        execute_eval_b(verbose=args.verbose)
    elif args.mode == "plots":
        execute_plots()
    elif args.mode == "slides":
        execute_slides()
    elif args.mode == "reproduce":
        execute_reproduce()
    elif args.mode == "full":
        execute_full()


if __name__ == "__main__":
    main()
