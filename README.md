# 🤖 AI-hayling  
🎯 **EMOHayling – Automatic Scoring for the Emotional Hayling Test**

> A dedicated assistant to automatically score verbal inhibition in classical and emotional contexts — powered by NLP, embeddings, and interactive rules.

---

## 📘 About this repository

This repository contains the full codebase of **EMOHayling**, developed during a 6-month research internship at **ICube Laboratory (GAIA platform, CNRS / University of Strasbourg)**, in collaboration with **LPC (Laboratory of Psychology and Cognition)**.

It automates the scoring of the **Hayling Sentence Completion Test** (including its **emotional version**) using:
- ⚙️ rule-based filters (e.g., proper noun, insanity, repetition),
- 🧠 semantic similarity thresholds,
- 🧪 and fine-tuned contextual models (e.g., LaBSE).

---

## 🎥 Demo

▶️ [Watch the demo video](https://drive.google.com/file/d/1b0xwnzLlJPP63nqE96t4f4y8nbGyo_e8/view?usp=sharing)

---

## ✨ Key Features

- ✅ Scoring pipeline using **Word2Vec**, **FastText**, **LaBSE**, and **XLM-R-MNLI**
- ✅ Threshold calibration using **Balanced Accuracy**
- ✅ Transparent rule-based scoring for interpretability
- ✅ Automatic scoring of patient responses from **PsychoPy logs**
- ✅ Interactive **PyQt5 GUI** with editable predictions
- ✅ Manual override of scores with retrainable model
- ✅ Synthetic data generation with **LLM agents**

---

## 📁 Project Structure

📦 AI-hayling/
┣ 📂app/ → GUI and scoring logic
┣ 📂models/ → Pretrained models and fine-tuned embeddings
┣ 📂rules/ → Rule-based penalty assignment
┣ 📂scripts/ → Scripts for preprocessing, scoring, evaluation
┣ 📂labse_emotion/ → Fine-tuned LaBSE model (emotion version)
┣ 📜README.md → Project documentation
┣ 📜requirements.txt → Python dependencies
┣ 📜LICENSE → CeCILL v2.1 License


---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/insafhamdi/AI-hayling.git
cd AI-hayling
2. Create and activate virtual environment

On Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate
On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

3. Install dependencies
pip install -r requirements.txt
