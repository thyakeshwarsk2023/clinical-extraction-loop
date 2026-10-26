"""
src/data_pipeline/adapters/ekacare_adapter.py
=============================================================================
EkaCare Medical Records Parsing Validation Set Adapter.
Provides parser for Hugging Face parquet / structured JSON format.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("BioTrace.EkaCareAdapter")


class EkaCareAdapter:
    """Standardized adapter for EkaCare Validation Set."""

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.raw_dir = os.path.join(data_root, "raw")

    def inspect_dataset(self) -> Dict[str, Any]:
        """Scans the downloaded EkaCare directory."""
        if not os.path.exists(self.raw_dir):
            return {
                "status": "directory_missing",
                "document_count": 0,
                "notes": "Folder data/external/ekacare/raw/ is empty. Hugging Face authentication required."
            }

        files = []
        for root, _, filenames in os.walk(self.raw_dir):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in [".png", ".jpg", ".jpeg", ".parquet", ".json"]:
                    files.append(os.path.join(root, fn))

        return {
            "status": "ready" if files else "empty",
            "document_count": len(files),
            "files": files,
            "supported_metrics": ["fhir_entity_recall", "clinical_key_value_f1", "llm_rubric_match"],
            "unsupported_metrics": ["character_level_cer (no char bbox annotations)"]
        }

    def generate_evaluation_manifest(self, output_path: str) -> Dict[str, Any]:
        """Creates locked evaluation manifest for EkaCare clinical records."""
        inspection = self.inspect_dataset()
        files = inspection.get("files", [])

        entries = []
        for f in files:
            doc_id = os.path.splitext(os.path.basename(f))[0]
            entries.append({
                "doc_id": doc_id,
                "file_path": f,
                "supported_metrics": ["clinical_entity_f1", "llm_judge_score"],
                "evaluation_reference": {
                    "source": "ekacare_validation_set"
                }
            })

        manifest = {
            "dataset_name": "EkaCare Medical Records Parsing Validation Set",
            "benchmark_track": "external_clinical_vlm_evaluation",
            "locked_test_set": True,
            "document_count": len(entries),
            "entries": entries
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(manifest, out_f, indent=2)

        logger.info("Saved EkaCare evaluation manifest to: %s", output_path)
        return manifest
