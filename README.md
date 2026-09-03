# Lab W3D1: Profile Inference on a Real GPU

### Predictions (by hand)

* **Qwen2.5-1.5B weights at fp16 (2 bytes each):** 3.0 GB
* **Qwen2.5-1.5B weights at int8 (1 byte each):** 1.5 GB
* **Resident VRAM at 512 context, fp16:** 4.5 GB
* **Resident VRAM at 4096 context, fp16:** 5.5 GB
* **Comparison:** 4096 context is larger by roughly 1.0 GB due to the expanding KV cache.
* **GPU utilisation during single-request decode:** 30 percent

## 2. Collected Data

### Matrix Profile

| Data Type | Context Length | Resident VRAM (GB) | GPU Util Mean (%) | Throughput (tok/s) |
|---|---|---|---|---|
| fp16 | 512 | 4.982 | 37.2 | 20.1 |
| fp16 | 2048 | 5.154 | 68.7 | 24.6 |
| fp16 | 4096 | 5.359 | 86.3 | 24.9 |
| int8 | 512 | 3.650 | 24.2 | 5.6 |
| int8 | 2048 | 3.879 | 26.9 | 5.2 |
| int8 | 4096 | 4.289 | 29.1 | 4.9 |

### Batch Scaling (fp16, 512 context)

* **Batch 1:** 28.7 tok/s, 48.7% GPU util, 4.982 GB VRAM
* **Batch 8:** 197.1 tok/s, 74.3% GPU util, 5.316 GB VRAM
* **Throughput Ratio:** 6.87x increase
* **Utilisation Delta:** +25.6%

## 3. Verification

`GREEN CHECK: PASS`

<img width="587" height="62" alt="Screenshot 2026-09-02 at 12 57 08 PM" src="https://github.com/user-attachments/assets/9acacea1-3ad6-4f2a-90b2-3fd5c04144c8" />


---

# Extra Lab W3D1: The Memory Leak Hunter

### Predictions (by hand)

* **Baseline after `unload()`:** `torch.cuda.memory_reserved()` is expected to return close to the pre-load baseline (within 0–50 MB) rather than exactly matching it, due to the PyTorch CUDA caching allocator holding onto reserved memory blocks.
* **Retaining outputs in a loop:** Appending output tensors without clearing them will cause resident VRAM to increase steadily with each iteration. This is **expected behaviour** (not a PyTorch bug) because user references keep both the activation tensors and their autograd computation graphs alive in GPU memory.


## 1. Baseline Control (Reload Loop)

Model: `Qwen/Qwen2.5-1.5B-Instruct` across 5 load/unload cycles.

| Cycle | After Load (MB) | After Unload (MB) |
| :---: | :---: | :---: |
| 0 | 3134.0 | 0.0 |
| 1 | 3134.0 | 0.0 |
| 2 | 3134.0 | 0.0 |
| 3 | 3134.0 | 0.0 |
| 4 | 3134.0 | 0.0 |


## 2. Leak Detection Summary

* **Leaky Run (no `torch.no_grad()`, retained logits):**
  * Slope: `211.248 MB/iteration`
  * Verdict: `Leaking = True`
* **Fixed Run (`torch.no_grad()`, immediate item extraction):**
  * Slope: `-0.0 MB/iteration`
  * Verdict: `Leaking = False`


## 3. Verification

`GREEN CHECK: PASS`

<img width="618" height="48" alt="Screenshot 2026-09-03 at 8 35 47 AM" src="https://github.com/user-attachments/assets/02b2fadc-f84a-4a09-a437-7b0172d59520" />


---


# Bug Lab W3D2: The Prompt That Wasn't as Long as You Asked

## 1. Overview
A bug investigation and fix for prompt-length generation and VRAM memory profiling.

- **Issue**: `prompt_of_len(n_tokens)` silently truncated prompts when requesting large token counts because Python list slicing past the list bounds returned fewer tokens without error.
- **VRAM Profiling Issue**: Memory wasn't fully reclaimed between model precision runs when unloaded inside a function helper due to lingering references. In-scope deletion and cache cleanup resolved the memory leak.


## 2. Results

| Precision | Before (Broken `unload()`) | After (In-Scope `del model`) |
| :--- | :--- | :--- |
| **fp16** | 4.19 GB | 3.06 GB |
| **int8** | 4.79 GB | 1.74 GB |
| **int4** | 2.88 GB | 1.15 GB |

**Memory Progression:** 1.15 GB (int4) < 1.74 GB (int8) < 3.06 GB (fp16)


## 3. Verification

`GREEN CHECK: PASS`

<img width="219" height="41" alt="Screenshot 2026-09-03 at 9 06 17 AM" src="https://github.com/user-attachments/assets/e3860abb-d157-4bc6-94ec-cd6c34800edc" />
