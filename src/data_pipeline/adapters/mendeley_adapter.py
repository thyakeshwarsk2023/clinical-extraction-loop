"""
src/data_pipeline/adapters/mendeley_adapter.py
=============================================================================
Mendeley Clinical Laboratory Test Reports Dataset Adapter.
DOI: 10.17632/bygfmk4rx9.2
Scans and normalizes real Egyptian clinical reports (260 documents) when downloaded.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("BioTrace.MendeleyAdapter")


class MendeleyAdapter:
    """Standardized adapter for Mendeley Clinical Laboratory Test Reports."""

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.raw_dir = os.path.join(data_root, "raw")

    def inspect_dataset(self) -> Dict[str, Any]:
        """Scans the downloaded Mendeley directory."""
        if not os.path.exists(self.raw_dir):
            return {
                "status": "directory_missing",
                "document_count": 0,
                "notes": "Folder data/external/mendeley/raw/ is empty. Manual download from Mendeley Data required."
            }

        files = []
        for root, _, filenames in os.walk(self.raw_dir):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in [".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff"]:
                    files.append(os.path.join(root, fn))

        return {
            "status": "ready" if files else "empty",
            "document_count": len(files),
            "files": files,
            "supported_metrics": ["end_to_end_layout_parsing", "field_name_recall", "unaligned_ocr_wer"],
            "unsupported_metrics": ["biochemical_invariants (uncurated)"]
        }

    def generate_evaluation_manifest(self, output_path: str) -> Dict[str, Any]:
        """Creates locked evaluation manifest for Mendeley reports."""
        inspection = self.inspect_dataset()
        files = inspection.get("files", [])

        entries = []
        for f in files:
            doc_id = os.path.splitext(os.path.basename(f))[0]
            entries.append({
                "doc_id": doc_id,
                "image_path": f,
                "supported_metrics": ["layout_parsing", "token_recall"],
                "requires_manual_curation_for_invariants": True,
                "evaluation_reference": {
                    "source": "mendeley_bygfmk4rx9"
                }
            })

        manifest = {
            "dataset_name": "Clinical Laboratory Test Reports (Mendeley)",
            "benchmark_track": "external_real_document_evaluation",
            "locked_test_set": True,
            "document_count": len(entries),
            "entries": entries
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(manifest, out_f, indent=2)

        logger.info("Saved Mendeley evaluation manifest to: %s", output_path)
        return manifest
