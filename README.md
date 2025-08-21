<div>
	<img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/DeepMind_new_logo.svg" alt="DeepMind Logo" width="180" style="margin: 0 40px 0 0;"/>
	<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face Logo" width="180"/>
	<img scr="saints logo" alt="Saints logo" width="180"/>
</div>

# DeepMind Financial NLP Explainability & Optimisation Suite (for Small Language Models)


**A research-grade, modular pipeline for explainability-driven fine-tuning, benchmarking, and deployment of small language models (SLMs) for financial sentiment analysis and classification.**


> **Focus:** This suite is specifically designed for *small language models* (e.g., TinyBERT, DistilBERT, MiniLM, MobileBERT) to enable efficient, interpretable, and production-ready NLP in financial domains. It includes tools for model compression, explainability, and robust benchmarking tailored to SLMs.


Built with [DeepMind](https://deepmind.com/) best practices and leveraging [Hugging Face Transformers](https://huggingface.co/transformers/).

<p>
	<img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
	<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
	<img src="https://img.shields.io/badge/huggingface-compatible-yellow" alt="Hugging Face Compatible">
</p>

## Table of Contents

- [Motivation & Background](#motivation--background)
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Results & Example Outputs](#results--example-outputs)
- [Acknowledgements](#acknowledgements)
- [License](#license)


## Motivation & Background

Financial NLP is a challenging domain where interpretability, efficiency, and robustness are critical. Large language models are often impractical for real-world deployment due to resource constraints and lack of transparency. This project addresses these challenges by providing a modular, explainability-driven pipeline for *small language models* (SLMs), enabling:

- Efficient, production-ready sentiment analysis and classification for financial text
- Transparent, explainable predictions for regulatory and business needs
- Rapid experimentation and benchmarking for research and industry

## Collaborators

- [Matthew Millward](https://github.com/MMillward2012) — DeepMind Intern, Durham University
- [Frank Soboczenski](https://github.com/h21k) — Research Collaborator, University of York
- [SAINTS](https://www.saints-cdt.ai) (York Centre for Security, Analytics, and Information Technology)
- Google DeepMind Research Ready Scheme

*If you contributed to this project and would like to be listed, please open a pull request or contact the maintainer.*

---


## Overview

This repository provides a **full-stack, research-oriented pipeline** for:

- **End-to-end data preparation**: Bring your own financial text data—automatic cleaning, validation, and stratified splitting are handled for you.
- **Config-driven setup**: All pipeline steps are controlled by a single, human-readable config file (`config/pipeline_config.json`), making experiments reproducible and easy to modify.
- **Model training**: Train your own Hugging Face transformer models (TinyBERT, DistilBERT, MiniLM, MobileBERT, FinBERT, and more) or PyTorch models, with adaptive hyperparameters and anti-overfitting safeguards.
- **ONNX export and quantization**: Seamlessly convert trained models to ONNX for fast inference and deployment. (Quantization support is being added for even better results.)
- **Explainability & error analysis**: Use the interactive dashboard to analyze model predictions with SHAP, LIME, attention, and GradCAM, and generate actionable insights for targeted fine-tuning.
- **Targeted fine-tuning**: Automatically focus training on misclassified and low-confidence samples, with multi-phase, weighted training and early stopping.
- **Production-style benchmarking**: Benchmark models for latency, throughput, and memory on CPU/GPU, with robust outlier filtering and hardware-aware optimization.
- **Model compression**: Apply structured pruning, quantization, and knowledge distillation to deploy truly efficient SLMs.
- **Interactive dashboards**: ipywidgets-based UIs for explainability, fine-tuning, and benchmarking.

## Safety & Explainability

This project is developed in collaboration with SAINTS (the York Centre for Security, Analytics, and Information Technology), with a strong emphasis on safety and trustworthiness in financial AI applications. By integrating advanced explainability tools (SHAP, LIME, attention, GradCAM), the pipeline enables:

- Transparent model decisions for regulatory and business requirements
- Identification and mitigation of model biases or errors
- Safer deployment of NLP models in sensitive financial contexts

Explainability is central to ensuring that model predictions are interpretable, auditable, and aligned with ethical standards—key priorities for both SAINTS and the broader financial AI community.

## Repository Structure

```
deepmind_internship/
│
├── config/                   # Pipeline and state configuration files
│   ├── pipeline_config.json
│   ├── pipeline_config_template.json
│   └── pipeline_state.json
│
├── data/                     # Raw and processed datasets
│   ├── FinancialAuditor/
│   ├── FinancialClassification/
│   ├── FinancialPhraseBank/
│   └── processed/
│
├── models/                   # All trained, fine-tuned, and compressed models
│   ├── all-MiniLM-L6-v2-financial-sentiment/
│   ├── distilbert-financial-sentiment/
│   ├── finbert-tone-financial-sentiment/
│   ├── mobilebert-uncased-financial-sentiment/
│   ├── tinybert-financial-classifier/
│   ├── tinybert-financial-classifier-fine-tuned/
│   ├── tinybert-financial-classifier-pruned/
│
├── notebooks/   			  # Generalised and advanced pipeline notebooks
│   ├── 0_setup.ipynb
│   ├── 1_data_processing.ipynb
│   ├── 2_train_models.ipynb
│   ├── 3_convert_to_onnx.ipynb
│   ├── 4_explainability.ipynb
│   ├── 5a_explainability_fine_tuning.ipynb
│   ├── 5b_fine_tune_comparison.ipynb
│   ├── 6_colab_benchmarks.ipynb
│   └── 6_benchmarks.ipynb
│
├── results/                  # Benchmarking and evaluation outputs
│   ├── benchmark_results.csv
│   └── ... (plots, summaries, reports)
│
├── src/                      # Core pipeline utilities and custom modules
│   ├── pipeline_utils.py
│   └── ...
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---



## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/MMillward2012/deepmind_internship.git
cd deepmind_internship
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your data
- Place your CSV or text data in the `data/` directory. The pipeline will automatically clean, validate, and split your data for you.

### 4. Configure your experiment
- Edit `config/pipeline_config.json` to select your model (e.g., TinyBERT, DistilBERT, MiniLM, MobileBERT, FinBERT), set hyperparameters, and choose data sources. No code changes required.

### 5. Train your model
- Run the training notebook: `notebooks/2_train_models.ipynb`.
- Supports Hugging Face and PyTorch models.

### 6. Convert to ONNX (and quantise)
- Use `notebooks/3_convert_to_onnx.ipynb` to export your model for fast inference. (Quantisation support coming soon.)

### 7. Run explainability and error analysis
- Open `notebooks/4_explainability.ipynb` to launch the interactive dashboard.
- Analyse errors, generate SHAP/LIME/attention/GradCAM explanations, and get recommendations for fine-tuning.

### 8. Fine-tune and benchmark
- Use `notebooks/5a_explainability_fine_tune.ipynb` for targeted, analysis-driven fine-tuning.
- Benchmark your models in `notebooks/6_colab_benchmarks.ipynb` or `6_benchmarks.ipynb` for latency, throughput, and memory.

### 9. Deploy or iterate
- Export your best models from the `models/` directory for deployment, or iterate with new configs and data.


**All steps are modular and can be run independently or as a full pipeline.**


## Results & Example Outputs

Below are real benchmarking results (batch size = 1, best latency) from the latest pipeline runs:

| Model                                   | Accuracy | F1 Score | Avg Latency (ms) | Model Size (MB) |
|-----------------------------------------|----------|----------|------------------|-----------------|
| TinyBERT (fine-tuned, ONNX+CUDA)        | 0.891    | 0.892    | 1.47             | 54.8            |
| TinyBERT (standard, ONNX+CUDA)          | 0.798    | 0.798    | 1.48             | 54.8            |
| MiniLM (all-MiniLM-L6-v2, ONNX+CUDA)    | 0.790    | 0.790    | 2.22             | 86.8            |
| DistilBERT (ONNX+CUDA)                  | 0.825    | 0.825    | 3.46             | 255.5           |
| MobileBERT (ONNX+CUDA)                  | 0.817    | 0.816    | 5.54             | 94.5            |
| FinBERT (ONNX+CUDA)                     | 0.839    | 0.839    | 220              | 419.0           |



This project is part of my internship for the **Google DeepMind Research Ready Scheme** at the University of York, in collaboration with **SAINTS** (the York Centre for Security, Analytics, and Information Technology).

Special thanks to my supervisors and mentors at DeepMind, the University of York, and SAINTS for their invaluable guidance and support, especially:
- Dr. Frank Soboczenski, University of York
- Dr. Ana Cavalcanti, University of York
- The SAINTS research group
- The other interns (Abdul, Anthony, Luke, Shehab, Sam, Maruf, Sky, Fernanda, James, Daniel and Aaron)

---

## Citation

If you use this repository or pipeline in your research, please cite as:

```bibtex
@misc{millward2025deepmindslm,
	title={DeepMind Financial NLP Explainability & Optimization Suite (for Small Language Models)},
	author={Matthew Millward},
	year={2025},
	howpublished={\url{https://github.com/your-org/deepmind_internship}}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Matthew Millward
