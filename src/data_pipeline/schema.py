"""
src/data_pipeline/schema.py
=============================================================================
BioTrace Dataset Expansion — Data Schemas and Annotation Protocols.
Provides strongly-typed schemas for:
  - Bounding boxes and analyte metadata
  - Artifact configurations and degradation parameters
  - Separate Benchmark Tracks (Track A: Image OCR vs Track B: OCR Output Correction)
  - Leakage-safe dataset splitting (Train, Val, Held-out Test)
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple


class TrackType(str, Enum):
    """
    Two strictly separated benchmark tracks:
      - TRACK_A_IMAGE_DEGRADATION: Document pixels are degraded (optical/physical),
        while printed text and ground truth labels correspond directly to what
        was intended to be rendered. Evaluates OCR robustness.
      - TRACK_B_OCR_CORRECTION: Controlled transcription errors are injected into
        the OCR output stream, while high-resolution or recoverable images
        remain available for independent visual grounding. Evaluates symbolic
        and physical micro-crop self-correction.
    """
    TRACK_A_IMAGE_DEGRADATION = "track_a_image_degradation"
    TRACK_B_OCR_CORRECTION = "track_b_ocr_correction"


class SplitType(str, Enum):
    """
    Leakage-safe split tiers partitioned strictly by underlying patient source ID
    and template family.
    """
    TRAIN = "train"
    VAL = "validation"
    TEST_SYNTHETIC = "test_synthetic"
    TEST_EXTERNAL_PUBLIC = "test_external_public"


class ArtifactType(str, Enum):
    """Supported optical, physical, and digital artifact categories."""
    CLEAN = "clean"
    BLUR_GAUSSIAN = "blur_gaussian"
    BLUR_MOTION = "blur_motion"
    BLUR_DEFOCUS = "blur_defocus"
    LOW_RESOLUTION = "low_resolution"
    JPEG_COMPRESSION = "jpeg_compression"
    NOISE_GAUSSIAN = "noise_gaussian"
    NOISE_SPECKLE = "noise_speckle"
    UNEVEN_LIGHTING = "uneven_lighting"
    SKEW_ROTATION = "skew_rotation"
    PERSPECTIVE_WARP = "perspective_warp"
    FAINT_INK = "faint_ink"
    CREASE_FOLD = "crease_fold"
    STAIN_SMUDGE = "stain_smudge"
    GRID_LINE_OVERLAY = "grid_line_overlay"
    DOT_MATRIX_RIBBON = "dot_matrix_ribbon"


class ArtifactSeverity(str, Enum):
    CLEAN = "clean"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class BoundingBox:
    """Exact pixel coordinates [x_min, y_min, x_max, y_max]."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_list(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def is_valid(self, width: int = 2000, height: int = 2000) -> bool:
        return (
            0 <= self.x_min < self.x_max <= width and
            0 <= self.y_min < self.y_max <= height
        )


@dataclass
class AnalyteItem:
    """Detailed metadata for a single analyte row in a laboratory report."""
    name: str
    key: str
    true_value: float
    rendered_value_str: str
    unit: str
    reference_interval: str
    value_bbox: BoundingBox
    label_bbox: Optional[BoundingBox] = None
    unit_bbox: Optional[BoundingBox] = None
    ref_bbox: Optional[BoundingBox] = None
    is_corrupted: bool = False
    injected_ocr_value: Optional[float] = None
    corruption_rule: Optional[str] = None


@dataclass
class ArtifactConfig:
    """Specification of an applied degradation artifact."""
    artifact_type: ArtifactType
    severity: ArtifactSeverity
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentAnnotation:
    """
    Comprehensive, self-contained annotation structure for an expanded
    clinical laboratory document.
    """
    doc_id: str
    source_id: str
    template_id: str
    panel_type: str
    track: TrackType
    split: SplitType
    method_text: str
    active_equation: str
    unit_system: str
    random_seed: int
    image_width: int
    image_height: int

    # Values & Analytes
    clean_values: Dict[str, float]
    printed_values: Dict[str, float]
    analytes: List[AnalyteItem]

    # Degradation specifications
    artifacts: List[ArtifactConfig]
    is_clean_control: bool
    is_ambiguous: bool
    requires_abstention: bool
    abstention_reason: Optional[str] = None

    # File paths (relative to dataset root)
    clean_image_path: str = ""
    degraded_image_path: str = ""
    metadata_json_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts annotation to serializable dictionary."""
        d = asdict(self)
        # Convert enums to string values
        d["track"] = self.track.value
        d["split"] = self.split.value
        d["artifacts"] = [
            {
                "artifact_type": a.artifact_type.value,
                "severity": a.severity.value,
                "parameters": a.parameters
            }
            for a in self.artifacts
        ]
        d["analytes"] = [
            {
                "name": a.name,
                "key": a.key,
                "true_value": a.true_value,
                "rendered_value_str": a.rendered_value_str,
                "unit": a.unit,
                "reference_interval": a.reference_interval,
                "value_bbox": a.value_bbox.to_list(),
                "label_bbox": a.label_bbox.to_list() if a.label_bbox else None,
                "unit_bbox": a.unit_bbox.to_list() if a.unit_bbox else None,
                "ref_bbox": a.ref_bbox.to_list() if a.ref_bbox else None,
                "is_corrupted": a.is_corrupted,
                "injected_ocr_value": a.injected_ocr_value,
                "corruption_rule": a.corruption_rule,
            }
            for a in self.analytes
        ]
        return d
