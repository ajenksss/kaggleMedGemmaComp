# 🏥 MedGemma: The Triage Copilot
**Kaggle MedGemma Impact Challenge Submission**

---

> **Disclaimer**: This project is a prototype clinical decision support demonstration. It is not medical advice. Outputs are suggestions for clinician review.

---

## 1. The 10-Second Pitch (The "Why")
**The Problem**: In the ICU, "Stable" is a lie. 
Patients often suffer from **Compensated Shock**—their blood pressure looks normal because their body is fighting to keep it up, but their physiology is secretly collapsing. By the time the alarms beep, it's too late.

**The Solution**: MedGemma TPT (Topological-Physical Twin).
It is a **Triage Copilot** that doesn't just read numbers; it reads *meaning*. 
It detects the "abnormal meaning" behind "normal numbers": specifically, the **Silent Instability** that precedes a crash.

**The Differentiator**:
> **"Threshold systems trigger on abnormal vitals. We trigger on abnormal meaning: normal numbers with unstable shape."**

---

## 2. The Use Case: The "Sarah" Scenario (Compensated Shock)
Meet Sarah. She is post-op.
*   **The Monitor**: BP 110/70. HR 85. **Green Light.** (Standard monitors see "Normal").
*   **The Reality**: Her stress response is maxed out. Her physiology is chaotic. She is 30 minutes away from Sepsis.
*   **MedGemma**: Detects "Topological Explosion" (High Variance) despite the stable BP.
    *   *Output*: **"Concern for Compensated Shock. Vitals stable, but physiological pattern is unstable. Suggest checking Lactate."**

**Impact**: We buy the doctor 60 minutes of "Golden Hour" time to intervene before the crash.

---

## 3. How It Works (The "Meaning Bridge")

We don't replace the doctor. We give them a **Copilot** that manages the cognitive load of "Ambiguity."

### Layer 1: The "Clinical Synthesizer" (MedGemma)
The Hero of our system. It is not just a chatbot. It is a **Reasoning Engine** that resolves conflict.
*   It takes conflicting signals ("BP is Good" vs "Math is Bad").
*   It applies **Sepsis-3 Guidelines** to synthesize a safe, structured rationale.
*   **Output**: NO DIAGNOSIS. Instead, it provides a **Risk State** (Red/Orange/Green) and **Clarifying Checks**.

### Layer 2: The "Evidence Stack" (TDA & Physics)
Hidden in the background until needed (Collapsible Evidence).
*   **Topological Sensor (TDA)**: Measures the "Shape" of the heartbeat. Exploring "Shape" detects instability that "Numbers" miss.
*   **Physics Guardrails (PINN)**: Uses Fluid Dynamics to veto false alarms. If a sensor glitches, Physics says "Impossible" and silences the alert.

---

## 4. The Architecture (Why we win $10k)
This is not a "Black Box". It is an **Explainable System**.
1.  **Safety First**: Physics Vetoes Hallucinations.
2.  **Privacy First**: Runs on Edge (WASM/Local).
3.  **Human First**: MedGemma speaks in "Provider-Ready" summaries, not raw math.

---

## 5. Deployment & Assets
*   **Immediate Utility**: Deployed as a "Shadow Monitor" on existing ICU tablets.
*   **Open Source Assets**:
    *   `src/layer_4_agent.py`: The Triage Copilot Logic.
    *   `data/medgemma_physics_distillation.jsonl`: 1,000 synthetic clinical scenarios for fine-tuning medical agents on "Physics-Verified" reasoning.

**MedGemma TPT: Turning Silence into Safety.**
