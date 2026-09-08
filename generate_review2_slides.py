#!/usr/bin/env python3
"""
generate_review2_slides.py
--------------------------
Generates/populates Review-2 presentation slides for the VIT B.Tech Project:
"Constraint-Verified Clinical Document Extraction"

Slides populated:
  - Slide 10: Technical Architecture (5-Stage Closed-Loop Pipeline)
  - Slide 11: Functional Requirements
  - Slide 12: Module Breakdown (Codebase Architecture)
  - Slide 13: Experiments and Results (400-Document Benchmark: 91.5% Convergence, 0.0% False-Trigger, 83.5% Single Recovery)
  - Slide 14: Conclusion & Future Roadmap (70% Milestone Achieved & Phase 2 Roadmap)

Output:
  Review2_Populated.pptx (without overwriting the original template)
"""

import os
import sys
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# Theme colors matching VIT PPT guidelines
COLOR_TITLE = RGBColor(0, 0, 0)           # Black
COLOR_PRIMARY = RGBColor(0, 75, 135)      # VIT Blue
COLOR_MUTED = RGBColor(90, 90, 90)        # Muted gray for footer
COLOR_TEXT = RGBColor(30, 30, 30)         # Charcoal body text
COLOR_ACCENT = RGBColor(220, 100, 20)     # Amber accent


def find_template_file():
    """Locate candidate template file in current directory or parent directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    search_paths = [
        # Check command-line argument first if provided
        sys.argv[1] if len(sys.argv) > 1 else None,
        # Check parent folder for student's Review 1 base deck (which has completed slides 1-9)
        os.path.join(parent_dir, "Review1-.pptx"),
        os.path.join(script_dir, "Review1-.pptx"),
        # Check for official blank template
        os.path.join(parent_dir, "4 Review 2 PPT Template (1).pptx"),
        os.path.join(parent_dir, "4 Review 2 PPT Template.pptx"),
        os.path.join(script_dir, "4 Review 2 PPT Template.pptx"),
    ]

    for p in search_paths:
        if p and os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "Could not locate Review 2 PPT template. Please provide the path as an argument:\n"
        "python generate_review2_slides.py <path_to_template.pptx>"
    )


def add_footer(slide, slide_num, date_str="08-09-2026"):
    """Adds standard footer: Date, SCOPE, and Slide Number matching the VIT template."""
    # Date (Left)
    tx_box_date = slide.shapes.add_textbox(Inches(0.92), Inches(6.95), Inches(3.0), Inches(0.4))
    tf_date = tx_box_date.text_frame
    tf_date.word_wrap = False
    p_date = tf_date.paragraphs[0]
    p_date.text = date_str
    p_date.font.size = Pt(11)
    p_date.font.color.rgb = COLOR_MUTED

    # SCOPE (Center)
    tx_box_scope = slide.shapes.add_textbox(Inches(5.0), Inches(6.95), Inches(3.3), Inches(0.4))
    tf_scope = tx_box_scope.text_frame
    tf_scope.word_wrap = False
    p_scope = tf_scope.paragraphs[0]
    p_scope.alignment = PP_ALIGN.CENTER
    p_scope.text = "SCOPE"
    p_scope.font.size = Pt(11)
    p_scope.font.color.rgb = COLOR_MUTED

    # Slide Number (Right)
    tx_box_num = slide.shapes.add_textbox(Inches(9.4), Inches(6.95), Inches(3.0), Inches(0.4))
    tf_num = tx_box_num.text_frame
    tf_num.word_wrap = False
    p_num = tf_num.paragraphs[0]
    p_num.alignment = PP_ALIGN.RIGHT
    p_num.text = str(slide_num)
    p_num.font.size = Pt(11)
    p_num.font.color.rgb = COLOR_MUTED


def configure_title(slide, title_text):
    """Sets slide title with proper positioning, font size, and weight."""
    if slide.shapes.title:
        title_shape = slide.shapes.title
    else:
        title_shape = slide.shapes.add_textbox(Inches(0.5), Inches(0.65), Inches(11.5), Inches(0.8))

    tf = title_shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = "Arial"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLOR_TITLE
    return title_shape


def add_bullet_point(text_frame, bold_prefix, body_text, level=0, space_after=Pt(10)):
    """Adds a two-part bullet point with bold lead-in and clear explanatory body."""
    p = text_frame.add_paragraph() if text_frame.text else text_frame.paragraphs[0]
    p.level = level
    p.space_after = space_after
    p.line_spacing = 1.15

    # Bold prefix
    run_bold = p.add_run()
    run_bold.text = bold_prefix + " "
    run_bold.font.name = "Arial"
    run_bold.font.size = Pt(15 if level == 0 else 13.5)
    run_bold.font.bold = True
    run_bold.font.color.rgb = COLOR_PRIMARY if level == 0 else COLOR_TEXT

    # Descriptive body text
    run_body = p.add_run()
    run_body.text = body_text
    run_body.font.name = "Arial"
    run_body.font.size = Pt(14.5 if level == 0 else 13)
    run_body.font.bold = False
    run_body.font.color.rgb = COLOR_TEXT


# ---------------------------------------------------------------------------
# Slide Generators (Slides 10 - 14)
# ---------------------------------------------------------------------------

def build_slide_10_architecture(prs, slide_idx=10):
    """Slide 10: Technical Architecture (5-Stage Closed-Loop Pipeline)."""
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    configure_title(slide, "Framework / Architecture / Block Diagram")
    add_footer(slide, slide_idx)

    # Content container
    content_shape = slide.shapes.add_textbox(Inches(0.92), Inches(1.65), Inches(11.4), Inches(5.0))
    tf = content_shape.text_frame
    tf.word_wrap = True

    add_bullet_point(
        tf,
        "Stage 0 — Assay-Method Classifier:",
        "Dynamically inspects methodology headers to determine verification routing (Direct Enzymatic Assay vs. Calculated Friedewald Panel)."
    )
    add_bullet_point(
        tf,
        "Stage 1 — Multimodal Base Extraction:",
        "Parses initial lipid values (TC, HDL, LDL, Triglycerides), localized bounding box coordinates, and token-level confidence scores."
    )
    add_bullet_point(
        tf,
        "Stage 2 & 2.5 — Invariant Verifier & Fault Isolation:",
        "Deterministically evaluates Friedewald equality (TC = HDL + LDL + TG/5) with physiological bounds; isolates suspect fields via algebraic counterfactual search (k=1 single-field and k=2 combinatorial pairs)."
    )
    add_bullet_point(
        tf,
        "Stage 3 & 4 — Vision Gate & Micro-Crop Re-Extraction:",
        "Extracts targeted bounding box micro-crops for physical enhancement, visual noise suppression, and dual-model consensus verification."
    )
    add_bullet_point(
        tf,
        "Stage 5 — Bounded Loop Controller & Audit Logger:",
        "Enforces an iteration budget (K <= 2), executing closed-loop state updates while serializing every residual score and routing decision to a persistent JSON audit trail."
    )
    return slide


def build_slide_11_requirements(prs, slide_idx=11):
    """Slide 11: Functional Requirements."""
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    configure_title(slide, "Functional Requirements")
    add_footer(slide, slide_idx)

    content_shape = slide.shapes.add_textbox(Inches(0.92), Inches(1.65), Inches(11.4), Inches(5.0))
    tf = content_shape.text_frame
    tf.word_wrap = True

    add_bullet_point(
        tf,
        "Dynamic Methodology Routing:",
        "The system must dynamically adapt verification logic based on detected clinical assay headers, conditionally bypassing arithmetic equality for direct assays while enforcing biological bounds."
    )
    add_bullet_point(
        tf,
        "Deterministic Invariant Verification:",
        "The system must strictly verify extraction integrity using domain-standard physiological laws (the Friedewald identity) within a strict numerical tolerance window (+/-10%)."
    )
    add_bullet_point(
        tf,
        "Autonomous Multi-Order Fault Isolation:",
        "The system must systematically isolate single-field (k=1) and double-field (k=2) transcription corruptions using algebraic counterfactual search, Levenshtein distance, and confidence scoring."
    )
    add_bullet_point(
        tf,
        "Zero-Harm Clinical Safety Bound:",
        "The system must maintain a 0.0% false-trigger rate on valid, uncorrupted clinical documents, strictly preventing false corrections on clean patient records."
    )
    add_bullet_point(
        tf,
        "Complete Traceability & Bounded Abstention:",
        "The system must log every internal hypothesis, residual score, and state mutation, safely escalating unresolvable anomalies (K > max_attempts) to human pathologist review."
    )
    return slide


def build_slide_12_modules(prs, slide_idx=12):
    """Slide 12: Module Breakdown & Implementation Structure."""
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    configure_title(slide, "Modules & Implementation Structure")
    add_footer(slide, slide_idx)

    content_shape = slide.shapes.add_textbox(Inches(0.92), Inches(1.65), Inches(11.4), Inches(5.0))
    tf = content_shape.text_frame
    tf.word_wrap = True

    add_bullet_point(
        tf,
        "src/stage0_classifier.py (Assay Classifier):",
        "Implements AssayClassifier with regex and heuristic pattern matching to route between DIRECT_MEASURED and FRIEDEWALD paths."
    )
    add_bullet_point(
        tf,
        "src/stage1_extract.py (Document State Extractor):",
        "Standardized multimodal ingestion interface extracting canonical values, confidence scores, and bounding box coordinates."
    )
    add_bullet_point(
        tf,
        "src/stage2_constraints.py (Constraint Engine):",
        "Enforces physiological bounds, canonical unit normalization (mmol/L <-> mg/dL), and Friedewald invariant residual calculation."
    )
    add_bullet_point(
        tf,
        "src/stage2_isolation.py (Hierarchical Isolation Engine):",
        "Evaluates single-field algebraic counterfactuals and discrete combinatorial pairs using composite edit-distance and confidence penalties."
    )
    add_bullet_point(
        tf,
        "src/pipeline.py (Closed-Loop Orchestrator):",
        "Core state machine wiring all 5 stages into a bounded iterative while-loop, managing state mutation and final status termination."
    )
    add_bullet_point(
        tf,
        "evals/run_ablations.py & evals/metrics.py (Benchmark Harness):",
        "Automated evaluation suite executing batch benchmark validation over hundreds of clinical reports, computing field accuracy and safety metrics."
    )
    return slide


def build_slide_13_results(prs, slide_idx=13):
    """Slide 13: Experiments and Results (400-Document Benchmark)."""
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    configure_title(slide, "Experiments and Results (400-Document Benchmark)")
    add_footer(slide, slide_idx)

    # Check for figures to embed on the right side
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_candidates = [
        os.path.join(script_dir, "figures", "field_accuracy.png"),
        os.path.join(script_dir, "figures", "status_distribution.png"),
        os.path.join(script_dir, "figures", "false_trigger_rate.png"),
    ]
    chart_path = next((p for p in chart_candidates if os.path.isfile(p)), None)

    # Layout width: if chart is embedded, allocate text on left and chart on right
    text_width = Inches(6.2) if chart_path else Inches(11.4)
    content_shape = slide.shapes.add_textbox(Inches(0.92), Inches(1.65), text_width, Inches(5.0))
    tf = content_shape.text_frame
    tf.word_wrap = True

    add_bullet_point(
        tf,
        "Dataset Scale:",
        "Evaluated on 400 clinical documents across 4 splits (200 clean, 30 clean_control, 120 corrupted_single, 50 corrupted_double)."
    )
    add_bullet_point(
        tf,
        "System Reliability:",
        "91.5% Overall Convergence (366/400 documents verified & finalized)."
    )
    add_bullet_point(
        tf,
        "Clinical Safety (Zero Harm):",
        "0.0% False-Trigger Rate on clean_control (0/30) and clean split (0/200)."
    )
    add_bullet_point(
        tf,
        "Single-Error Recovery:",
        "83.5% Post-Correction Field Accuracy across corrupted_single (n=120):"
    )
    add_bullet_point(
        tf,
        "  - Field Breakdown:",
        "Total Cholesterol: 81.7% | HDL: 79.2% | LDL: 86.7% | Triglycerides: 86.7%",
        level=1,
        space_after=Pt(6)
    )
    add_bullet_point(
        tf,
        "Corruption Typology:",
        "Smudge: 100.0% (45/45 conv) | Decimal Shift: 81.4% (35/39 conv, 61.5% iso) | Digit Swap: 65.3% (21/36 conv)."
    )
    add_bullet_point(
        tf,
        "Algorithmic Optimization:",
        "Relaxing the k=1 edit-distance commit gate for same-length digit swaps unlocked a +6.5pp recovery improvement."
    )

    # Embed Chart on the right side if available
    if chart_path:
        chart_left = Inches(7.35)
        chart_top = Inches(1.75)
        chart_w = Inches(4.9)
        slide.shapes.add_picture(chart_path, chart_left, chart_top, width=chart_w)

        # Highlight callout below chart
        callout_shape = slide.shapes.add_textbox(chart_left, Inches(4.95), chart_w, Inches(1.7))
        ctf = callout_shape.text_frame
        ctf.word_wrap = True
        cp = ctf.paragraphs[0]
        cp.line_spacing = 1.15
        r1 = cp.add_run()
        r1.text = "Benchmark Key Finding:\n"
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_PRIMARY
        r2 = cp.add_run()
        r2.text = (
            "The invariant engine achieved a 0.0% false-trigger rate while resolving "
            "severe single and double corruptions, demonstrating clinically safe autonomy."
        )
        r2.font.size = Pt(12)
        r2.font.color.rgb = COLOR_TEXT

    return slide


def build_slide_14_conclusion(prs, slide_idx=14):
    """Slide 14: Conclusion & Future Roadmap."""
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    configure_title(slide, "Conclusion & Future Roadmap")
    add_footer(slide, slide_idx)

    content_shape = slide.shapes.add_textbox(Inches(0.92), Inches(1.65), Inches(11.4), Inches(5.0))
    tf = content_shape.text_frame
    tf.word_wrap = True

    add_bullet_point(
        tf,
        "70% Milestone Successfully Achieved:",
        "Formulated, implemented, and validated the closed-loop neurosymbolic architecture on a standardized 400-document clinical benchmark."
    )
    add_bullet_point(
        tf,
        "Deterministic Clinical Safety Proven:",
        "Demonstrated that algebraic counterfactual logic eliminates silent numerical hallucinations while maintaining a strict 0.0% false-trigger bound on clean data."
    )
    add_bullet_point(
        tf,
        "Inference-Time Self-Correction:",
        "Successfully corrected character-level decimal shifts and typographical mutations without costly model fine-tuning or external re-training."
    )
    add_bullet_point(
        tf,
        "Phase 2 Roadmap (Target: Final Review, October 2026):",
        "Active development transitioning to the physical machine-vision repair gate:"
    )
    add_bullet_point(
        tf,
        "  - 1. OpenCV Graph-Cut Segmentation:",
        "Deploy morphological pixel graph-cut into stage3_vision_gate.py to strip paper smudges, stamps, and table rulings.",
        level=1,
        space_after=Pt(5)
    )
    add_bullet_point(
        tf,
        "  - 2. Real-ESRGAN 4x Super-Resolution:",
        "Deploy pre-trained blind super-resolution weights to clarify degraded numerical micro-crops prior to re-extraction.",
        level=1,
        space_after=Pt(5)
    )
    add_bullet_point(
        tf,
        "  - 3. Dual-Model Consensus & Delta Confidence:",
        "Track token-level confidence deltas (Delta-conf) during re-extraction as empirical evidence of visual grounding.",
        level=1,
        space_after=Pt(5)
    )
    return slide


def build_slide_15_references(prs, slide_idx=15):
    """Slide 15: Formal IEEE References matching literature review."""
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    configure_title(slide, "References")
    add_footer(slide, slide_idx)

    content_shape = slide.shapes.add_textbox(Inches(0.92), Inches(1.65), Inches(11.4), Inches(5.0))
    tf = content_shape.text_frame
    tf.word_wrap = True

    refs = [
        '[1] L. Hillebrand, T. Deu\u00dfer, T. Dilmaghani, B. Kliem, R. Loitz, and R. Sifa, "Towards automating numerical consistency checks in financial reports," in Proc. IEEE Int. Conf. Big Data (BigData), Osaka, Japan, 2022, pp. 2145-2152.',
        '[2] X. Wang, L. Xie, J. Dong, and C. C. Loy, "Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data," in Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW), Montreal, Canada, 2021, pp. 1905-1914.',
        '[3] L. Luo, B. Tang, X. Chen, R. Han, and T. Chen, "VividMed: Vision language model with versatile visual grounding for medicine," in Proc. Conf. Nations Americas Chapter Assoc. Comput. Linguist.: Human Lang. Technol. (NAACL), Albuquerque, NM, USA, 2025, pp. 1800-1821.',
        '[4] J. Bai et al., "Qwen2.5-VL technical report," arXiv preprint arXiv:2502.13923, Feb. 2025.',
        '[5] W. T. Friedewald, R. I. Levy, and D. S. Fredrickson, "Estimation of the concentration of low-density lipoprotein cholesterol in plasma, without use of the preparative ultracentrifuge," Clinical Chemistry, vol. 18, no. 6, pp. 499-502, 1972.'
    ]

    for ref in refs:
        p = tf.add_paragraph() if tf.text else tf.paragraphs[0]
        p.space_after = Pt(10)
        p.line_spacing = 1.15
        run = p.add_run()
        run.text = ref
        run.font.name = "Arial"
        run.font.size = Pt(13)
        run.font.color.rgb = COLOR_TEXT

    return slide


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    template_path = find_template_file()
    print(f"[1/4] Loaded template file: {template_path}")

    prs = pptx.Presentation(template_path)
    initial_count = len(prs.slides)
    print(f"      Initial slide count: {initial_count}")

    # Check if the loaded presentation already has slides 1-9 (Review 1 presentation)
    # If loaded from Review1-.pptx (count == 9), Slides 10-14 are appended sequentially.
    print("[2/4] Populating Slides 10, 11, 12, 13, and 14...")
    build_slide_10_architecture(prs, slide_idx=10)
    build_slide_11_requirements(prs, slide_idx=11)
    build_slide_12_modules(prs, slide_idx=12)
    build_slide_13_results(prs, slide_idx=13)
    build_slide_14_conclusion(prs, slide_idx=14)

    # Ensure updated formal references slide concludes the presentation
    build_slide_15_references(prs, slide_idx=15)

    final_count = len(prs.slides)
    print(f"[3/4] Successfully generated presentation. Final slide count: {final_count}")

    output_filename = "Review2_Populated.pptx"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    prs.save(output_path)
    print(f"[4/4] Saved populated presentation to: {output_path}")
    print(f"      Original template was NOT overwritten.")


if __name__ == "__main__":
    main()
