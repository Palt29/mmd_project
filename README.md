# Latent-Space Recommender System

---

## Index
1. [Overview](#overview)  
2. [Key Information](#key-information)  
3. [Repository Structure](#repository-structure)  
4. [Setup & Execution](#setup--execution)  
5. [Code Quality](#code-quality)  
6. [License](#license)

---

## Overview

This project was developed as part of the *Mining Massive Datasets* course at the University of Verona (A.Y. 2024/2025).  
It focuses on implementing and evaluating **latent-factor recommender systems** using the *Global and Local Singular Value Decomposition* (GLSVD) framework.

The main goal is to compare **latent-space models** (rGLSVD and sGLSVD) against classical **collaborative filtering** methods to assess whether the increased model complexity yields measurable performance improvements in top-N recommendation tasks.

The implementation was inspired by:

> Evangelia Christakopoulou and George Karypis,  
> *Local Latent Space Models for Top-N Recommendation*,  
> Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018.  
> https://doi.org/10.1145/3219819.3220112

---

## Key Information

- **Course:** Mining Massive Datasets (University of Verona)  
- **Academic Year:** 2024/2025  
- **Main Topics:** Collaborative Filtering, Matrix Factorization, Latent-Space Modeling, Recommendation Metrics  
- **Dataset:** MovieLens Latest Small  
- **Evaluation Metrics:** Hit Rate (HR) and Average Reciprocal Hit Rank (ARHR)  
- **Authors:** Andrea Arragoni & Pedro Alonso Lopez Torres  

---

## Repository Structure

| Path | Description |
|------|--------------|
| **.vscode/** | Editor configuration folder (not relevant to the implementation). |
| **data/** | Contains the CSV files representing the MovieLens dataset used for training and evaluation. |
| **docs/** | Documentation folder containing:<br>• `mmd_project_report.pdf`: final project report summarizing the methodology, results, and discussion.<br>• `local_latent_space_models_paper.pdf`: reference paper *Local Latent Space Models for Top-N Recommendation* by Christakopoulou and Karypis. |
| **src/** | Contains all source code:<br>• `collaborative_class.py`: user-user collaborative filtering implementation.<br>• `rglsvd_class.py`: implementation of rGLSVD (rank-varying).<br>• `sglsvd_class.py`: implementation of sGLSVD (subset-varying).<br>• `utils.py`: helper functions for data loading, preprocessing, and evaluation. |
| **.gitignore**, **.gitattributes** | Version control configuration files. |
| **.pre-commit-config.yaml** | Defines pre-commit hooks for Ruff and Mypy. |
| **.python-version** | Specifies the Python version used. |
| **3-recommendation.ipynb** | Original course notebook introducing collaborative filtering, later extended to integrate latent-factor models. It displays actual outputs for clarity. |
| **clusters.npy** | Precomputed user cluster assignments. |
| **main.py** | Script executing the complete training and evaluation pipeline for rGLSVD and sGLSVD. |
| **parameters_research.ipynb** | Notebook used for parameter tuning (number of clusters, global and local latent factors). |
| **pyproject.toml**, **uv.lock** | Project dependencies and environment configuration. |
| **utility_matrix.npy** | Precomputed user-item utility matrix. |

---

## Setup & Execution

This project uses **[uv](https://docs.astral.sh/uv/)** for environment management and reproducibility.

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/mmd_project.git
cd mmd_project
```

### 2. Set up the environment

```bash
uv sync
```

### 3. Run the main experiment

```bash
uv run python src/main.py
```

**Notes:**
- Python version is fixed in `.python-version` for reproducibility.  
- Requires **Python ≥ 3.12**.  
- Follow the official guide to install `uv`:  
  https://docs.astral.sh/uv/getting-started/installation

---

## Code Quality

Static analysis and style checking are handled using:

- **Ruff** — for linting and formatting  
- **Mypy** — for static type checking  

Both tools are automatically executed via pre-commit hooks defined in `.pre-commit-config.yaml`.