# Future Work: Real Vision-Language Model Baseline (Local VLM Probe)

## 1. Hardware Audit & Status
- **Date:** 2026-10-25
- **Tool:** `nvidia-smi`
- **Detected GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
- **Detected VRAM:** `4096 MiB` (4.0 GB)
- **PyTorch CUDA Status:** CPU-only binary installed (`torch.cuda.is_available() == False`)
- **Requirement:** VRAM $\ge$ 6.0 GB for quantized local VLM execution (e.g. Qwen2.5-VL 3B 4-bit AWQ/bitsandbytes with vision encoder feature maps and KV cache context).
- **Status:** **DEFERRED (Hardware Incompatible)**

---

## 2. Intended Model & Architecture
- **Model:** `Qwen/Qwen2.5-VL-3B-Instruct` (or 4-bit quantized AWQ/GPTQ/bitsandbytes derivative).
- **Target Pipeline Integration:** 
  - **Arm 0 (Primary Extractor):** Zero-shot document vision-language extraction returning structured JSON:
    ```json
    {
      "total_cholesterol": float,
      "hdl": float,
      "ldl": float,
      "triglycerides": float
    }
    ```
  - **Tier-2 Escalation Engine:** Conditional fallback triggered when Stage 2 constraint violations or optical noise (GrabCut contrast $< 0.40$) occur.
- **Evaluation Target:** Zero-shot field accuracy across sample test lab reports against ground-truth sidecar JSON.

---

## 3. Hardware Requirements for Deployment
To execute `Qwen2.5-VL-3B` locally without Out-Of-Memory (OOM) faults:
1. **Minimum GPU VRAM:** $\ge 6.0\text{ GB}$ (dedicated, unshared VRAM) for 4-bit quantized inference (e.g., RTX 3060 12GB, RTX 4060 Ti 16GB, or A10G/T4 cloud instances).
2. **Recommended GPU VRAM:** $\ge 8.0\text{ GB}$ to support native image token resolution without downscaling patches.
3. **Software Stack:** CUDA-enabled PyTorch build (`cu121` or `cu124`), `bitsandbytes` (or `vllm`/`autoawq`), and `transformers >= 4.49.0`.

---

## 4. Current Architecture Decision
Because available physical VRAM is 4.0 GB (< 6.0 GB threshold), local VLM weight loading is skipped per instructions. The pipeline maintains its deterministic Stage 1 extractor and Stage 2 constraint engine without loading non-functional heavy model weights into host swap memory.
