"""
src/data_pipeline/adapters/__init__.py
=============================================================================
External Dataset Adapters for BioTrace.
"""

from src.data_pipeline.adapters.clinocr_adapter import ClinOCRAdapter
from src.data_pipeline.adapters.mendeley_adapter import MendeleyAdapter
from src.data_pipeline.adapters.ekacare_adapter import EkaCareAdapter

__all__ = ["ClinOCRAdapter", "MendeleyAdapter", "EkaCareAdapter"]
