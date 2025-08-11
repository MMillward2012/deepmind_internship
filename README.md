

<div align="center">
	<img src="https://upload.wikimedia.org/wikipedia/commons/6/6e/DeepMind_logo.png" alt="DeepMind Logo" width="220" style="margin: 0 40px 0 0;"/>
	<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face Logo" width="220"/>
</div>

# DeepMind Financial NLP Explainability & Optimization Suite

A research-grade, modular pipeline for **explainability-driven fine-tuning, benchmarking, and deployment** of transformer models for financial sentiment analysis and classification.  
Built with [DeepMind](https://deepmind.com/) best practices and leveraging [Hugging Face Transformers](https://huggingface.co/transformers/).

---

## Team & Contributors

- Matthew Millward (DeepMind)
- Frank
- Other Interns

---

## Overview

This repository provides a full-stack workflow for:
- **Explainability analysis** (SHAP, LIME, attention, GradCAM)
- **Targeted fine-tuning** using explainability insights
- **Adaptive hyperparameter optimization**
- **Model compression (pruning, quantization, distillation)**
- **Production-style benchmarking (latency, throughput, accuracy)**
- **Interactive dashboards for research and reporting**

It is designed for robust, reproducible research and real-world deployment in financial NLP.

---

## Repository Structure

```
deepmind_internship/
│
├── notebooks/
│   ├── 5_explainability_generalized.ipynb      # Explainability analysis & dashboard
│   ├── 6_fine_tune_backup.ipynb                # Analysis-driven fine-tuning & compression
│   ├── 4_colab_benchmarks.ipynb                # Advanced benchmarking (latency, throughput, ONNX)
│   └── ...                                     # Additional research notebooks
│
├── models/                                     # All trained, fine-tuned, and compressed models
│
├── data/                                       # Raw and processed datasets
│
├── analysis_results/                           # Explainability and error analysis outputs
│
├── results/                                    # Benchmarking and evaluation outputs
│
├── src/                                        # Core pipeline utilities and custom modules
│
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/your-org/deepmind_internship.git
cd deepmind_internship
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the explainability dashboard
Open `notebooks/5_explainability_generalized.ipynb` in Jupyter or VS Code and run all cells.  
- Analyze model errors with SHAP, LIME, attention, and GradCAM.
- Generate actionable recommendations for fine-tuning and pruning.

### 4. Fine-tune with explainability guidance
Open `notebooks/6_fine_tune_backup.ipynb` and follow the workflow:
- Loads analysis results and adapts training strategy.
- Prevents overfitting and leakage with smart data handling.
- Supports model compression (pruning, quantization, distillation).

### 5. Benchmark for latency and efficiency
Open `notebooks/4_colab_benchmarks.ipynb`:
- Benchmarks PyTorch and ONNX models on CPU/GPU.
- Measures latency, throughput, memory, and accuracy.
- Compares before/after fine-tuning and compression.

---

## Key Features

- **Explainability-Driven Optimization:**  
	Uses SHAP/LIME/attention to guide data augmentation, sample weighting, and hyperparameter selection.

- **Adaptive Hyperparameters:**  
	Learning rate, batch size, and epochs are set dynamically based on model error patterns and complexity.

- **Anti-Overfitting & Data Safety:**  
	Stratified splits, overlap detection, and sample weighting to prevent leakage and bias.

- **Model Compression:**  
	Structured pruning, quantization (INT8), and knowledge distillation for real speedup.

- **Production-Style Benchmarking:**  
	End-to-end latency, throughput, and memory profiling with robust outlier filtering and hardware-aware optimization.

- **Interactive Dashboards:**  
	ipywidgets-based UIs for explainability, fine-tuning, and benchmarking.

---

## Example Usage

### Explainability Dashboard
- Analyze any model’s errors and attributions interactively.
- Identify problematic classes, tokens, and patterns.

### Fine-Tuning Pipeline
- Loads recommendations from explainability analysis.
- Runs multi-phase, weighted training with early stopping and learning rate scheduling.
- Exports models for benchmarking and deployment.

### Benchmarking
- Measures latency (mean, p95, p99), throughput, and memory.
- Compares PyTorch and ONNX models, with and without quantization.
- Reports efficiency metrics and exports results as CSV/JSON/plots.

---

## Acknowledgements

This project is inspired by DeepMind’s research standards and leverages the Hugging Face Transformers ecosystem.  
Special thanks to the open-source community and contributors to SHAP, LIME, Hugging Face, and PyTorch.

---

## Contributors

- [Your Name] (DeepMind)
- [Collaborators, if any]
- [Hugging Face Community]

---

## License

This repository is for research and educational use.  
See `LICENSE` for details.

---

<p align="center">
	<img width="200" src="https://upload.wikimedia.org/wikipedia/commons/6/6e/DeepMind_logo.png" alt="DeepMind Logo" style="margin: 0 30px 0 0;">
	<img width="200" src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face Logo">
</p>
