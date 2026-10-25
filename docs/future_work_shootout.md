# Future Work: Multi-Model Architecture-Agnostic Shootout (CLAIM-13)

## 1. Specification & Scope
- **Claim:** Benchmark clinical extraction gains across diverse multimodal architectures including LayoutLMv3, Donut (Document Understanding Transformer), and Qwen2.5-VL to demonstrate architecture-agnostic neurosymbolic error correction.
- **Current Status:** `not-implemented` (Deferred).
- **Hardware Constraint:** The host system possesses 4.0 GB VRAM (RTX 3050 Laptop GPU), which is below the 6.0 GB minimum threshold required for concurrent multi-model VLM weights, spatial grid cross-attention feature maps, and autoregressive decoding.

---

## 2. Planned Architecture Integration
When appropriate GPU hardware (>=16 GB VRAM) is provisioned, the multi-model shootout will evaluate three distinct model paradigms behind the `OCREngine` protocol:
1. **LayoutLMv3 (Discriminative Multimodal Transformer):** Token-level text + layout 2D bounding boxes + visual patch embeddings.
2. **Donut (OCR-Free Generative Encoder-Decoder):** Swin Transformer visual encoder with mBART generative text decoder producing raw clinical JSON without an external OCR engine.
3. **Qwen2.5-VL-7B / 3B (Modern Multimodal Large Language Model):** Dynamic resolution vision encoder with multi-head attention capable of structured clinical report extraction and reasoning.

---

## 3. Evaluation Protocol
Each architecture will be evaluated as Arm 0 feed-forward extractor across the 170-document ablation benchmark (`dataset_A_synthetic` and `dataset_B_realistic`):
- Compute initial Arm 0 Field Accuracy across all 4 lipid analytes.
- Pass predictions through Stage 2 Invariant Engine and Stage 2.5 Damerau-Levenshtein Fault Localization.
- Trigger Stage 3 GrabCut and Stage 4 Real-ESRGAN micro-crop repair on isolated suspect cells.
- Measure delta Field Accuracy and error reduction percentage to prove that closed-loop symbolic verification provides architecture-agnostic gains regardless of base visual model.
