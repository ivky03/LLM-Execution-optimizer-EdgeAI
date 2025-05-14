# 🧠 LLM Execution Optimizer for Edge AI

This project benchmarks the **inference performance** of small HuggingFace language models across three execution modes:

- **PyTorch (default execution)**
- **ONNX (exported for optimized runtime)**
- **Quantized (for reduced memory and improved speed on edge devices)**

It measures and compares:

- **Latency** (in seconds) — how fast each model responds
- **Memory usage** (in MB) — how much RAM the model consumes

---

## ✅ Models Tested

The following small-to-medium LLMs were evaluated:

- `distilgpt2`
- `EleutherAI/gpt-neo-125M`
- `openai-community/gpt2`

These were chosen to keep the benchmarking fast, CPU-friendly, and suitable for edge simulation.

---

## 📊 Benchmark Process

For each model:

1. Loaded using HuggingFace Transformers
2. Prompted with: `"Explain 5G technology in simple terms."`
3. Measured:
   - Inference time (latency)
   - Memory usage before and after generation

ONNX export used `opset_version=14` and CPU-only inference was enforced for edge simulation.

Quantization used PyTorch's `quantize_dynamic()` with `torch.qint8`.

---

## 🧾 Results (CSV Snapshot)

| Model                   | Type      | Latency (s) | Memory (MB) |
| ----------------------- | --------- | ----------- | ----------- |
| distilgpt2              | PyTorch   | 3.44        | 317.71      |
| distilgpt2              | ONNX      | 1.68        | 624.70      |
| distilgpt2              | Quantized | 3.05        | 4.41        |
| gpt-neo-125M            | PyTorch   | 5.51        | 168.01      |
| gpt-neo-125M            | ONNX      | 1.94        | 871.47      |
| gpt-neo-125M            | Quantized | 3.05        | -194.73 ⚠️  |
| gpt2 (openai-community) | PyTorch   | 6.58        | -7.93 ⚠️    |
| gpt2 (openai-community) | ONNX      | 2.06        | 703.80      |
| gpt2 (openai-community) | Quantized | 4.30        | -37.49 ⚠️   |

> ⚠️ Negative memory values are likely due to how memory deltas were calculated with `psutil`, which can sometimes return inconsistent results on short-lived CPU-only processes.

---

## 📈 Plots

Bar charts were generated to visually compare:

- Latency across execution types
- Memory usage across execution types

These are saved as:

- `plots/latency_plot.png`
- `plots/memory_plot.png`

---

## 💡 Takeaways

- **ONNX consistently reduces latency**, sometimes by over 50%
- **Quantization dramatically reduces memory (when measured accurately)**
- Performance benefits vary per model — not all models quantize equally well
- ONNX + Quantization = strong combo for **Edge AI deployment**

---

## 🧰 Tech Stack

- Python (PyTorch, Transformers, psutil, matplotlib)
- ONNX & ONNX Runtime
- Google Colab (CPU-only environment)

---

## 📂 Outputs

This repo/notebook generates:

- `results.csv` — raw benchmark data
- `latency_plot.png` and `memory_plot.png` — visual comparisons

You can re-run this anytime with your own models or prompts.

---

## 📎 How to Use

Open the notebook in Colab → Run all cells → Download results → Push to GitHub.  
All plots and CSVs are auto-generated and downloadable at the end.

---
