# MedGemma: Topological-Physical Twin (TPT)
**The "Physics-Gated" ICU Monitor | Kaggle MedGemma Impact Challenge**

---

> **Disclaimer**: This project is a prototype clinical decision support demonstration for the MedGemma Impact Challenge. It is not medical advice and must not be used for real patient care. Outputs are suggestions for clinician review and verification.

---

## 🏗️ Project Status: PROTOTYPE
This repository contains a **Code-Complete Prototype** of the TPT Architecture.

### ✅ What is Implemented (Code-Complete Prototype)
1.  **Topological Sensor (`src/layer_1_tda.py`)**:
    *   Uses actual **Johnson-Lindenstrauss Projection** (`sklearn`).
    *   Uses actual **Vietoris-Rips Persistence** (`gtda`) to calculate Homology.
    *   The "Shape Scores" are mathematically derived from the data geometry.
2.  **Physics Engine (`src/layer_2_pinn.py`)**:
    *   Implements real **Navier-Stokes/Ohm's Law** for Hemodynamics.
    *   The "Validity Checks" are based on strict medical physics constraints.
3.  **Synthetic Data Generator (`generate_finetune_data.py`)**:
    *   Produces mathematically consistent "Physics-Distilled" training data.

### ⚠️ Where MedGemma Fits (Prototype Note)
In this prototype, MedGemma is represented as the final reasoning layer (Layer 4). For Kaggle reproducibility, the demo uses deterministic clinical templates, but the architecture is designed for MedGemma to generate patient-facing summaries and clinician-facing explanations after physics and topology checks pass.

### 🚧 What is SIMULATED (For Demo Speed)
1.  **MedGemma Inference (`src/layer_4_agent.py`)**:
    *   To ensure 100Hz real-time performance in the demo, we simulate the *output* of the MedGemma 2B model using deterministic logic.
    *   **See `src/medgemma_proof.py`**: A reference implementation showing the ACTUAL code to run the model with our prompt structure (requires GPU).
    *   **Auto-Trigger Mode**: The App includes a toggle to "Auto-Run" the real model when Sepsis Risk > 80%, simulating an Edge-AI event loop.
2.  **Patient Data (`data/mock_vitals.csv`)**:
    *   Generated using a Gaussian Process to mimic Sepsis onset. Not real MIMIC-III patient records (for privacy/competition rules).

---

## 🚀 Quick Start
```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Generate Data
python src/generate_mock_data.py

# 3. Run the Dashboard
streamlit run src/app.py
```

## 📂 File Structure
*   `src/app.py`: The Main Streamlit Dashboard.
*   `src/layer_*.py`: The 4 Core Intelligence Layers.
*   `data/medgemma_physics_distillation.jsonl`: The Synthetic Training Corpus.

---

**"We don't need a smarter AI. We need a stricter one."**
