# Future Work: VLM Cross-Attention Dynamic Localization Hooks (CLAIM-14)

## 1. Specification & Scope
- **Claim:** Dynamically localize suspect character bounding boxes directly from VLM cross-attention token-to-patch attention maps (OCR heads), replacing static layout metadata bounding boxes.
- **Current Status:** `not-implemented` (Deferred).
- **Hardware Constraint:** Extracting and storing full multi-head cross-attention tensors $[B, H, T, P]$ (batch size $B$, attention heads $H$, generated token sequence length $T$, vision patches $P$) requires significant memory bandwidth and VRAM exceeding the 4.0 GB available on this machine.

---

## 2. Theoretical Architecture
In modern Vision-Language Models (e.g. Qwen2-VL, LLaVA-NeXT, PaLI-3):
1. **Forward Hook Registration:** Register PyTorch forward hooks on cross-attention layers in the decoder blocks:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
2. **Token-to-Patch Mapping:** For any generated analyte numeric token $t_i$ (e.g., `"211.0"`), extract the cross-attention weights over image visual patch embeddings $p_j$.
3. **Heatmap Thresholding & Contour Bounding:** Reshape the 1D patch attention vector into the 2D image grid $[H_p, W_p]$, interpolate to original image coordinates $[H, W]$, apply Otsu thresholding, and compute the minimum bounding box enclosing the maximal activation contour.
4. **Handoff to Machine Vision Gate:** Feed the dynamically extracted bounding box $[x_0, y_0, x_1, y_1]$ into Stage 3 GrabCut segmentation and Stage 4 Real-ESRGAN super-resolution.

---

## 3. Hardware Requirements
- **Minimum VRAM:** >= 12 GB VRAM (e.g., RTX 3060 12GB, RTX 4070, or A10G) to store intermediate cross-attention gradients/activations without paging to host RAM.
- **Software Dependencies:** `transformers >= 4.49.0`, `torch >= 2.2.0` with CUDA support.
