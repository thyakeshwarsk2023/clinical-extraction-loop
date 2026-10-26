"""
src/data_pipeline/__init__.py
=============================================================================
BioTrace Dataset Expansion Package.
Deterministic, multi-template, leakage-safe synthetic generator and benchmark suite.
"""

from src.data_pipeline.schema import (
    TrackType,
    SplitType,
    ArtifactType,
    ArtifactSeverity,
    BoundingBox,
    AnalyteItem,
    ArtifactConfig,
    DocumentAnnotation,
)

__all__ = [
    "TrackType",
    "SplitType",
    "ArtifactType",
    "ArtifactSeverity",
    "BoundingBox",
    "AnalyteItem",
    "ArtifactConfig",
    "DocumentAnnotation",
]
