# 🧠 AI-hayling

**🎯 EMOHayling – Automatic Scoring for the Emotional Hayling Test**

A clinical AI tool designed to automatically score verbal inhibition in classical and emotional contexts — powered by NLP, embeddings, and transformer models.


[![ICTAI 2025](https://img.shields.io/badge/Accepted--paper-ICTAI%202025-blue?logo=ieee)]([https://ictai-conference.org/](https://easyconferences.eu/ictai2025/))

---

## 📌 Project Overview

AI‑Hayling is a research project developed during a 6-month internship at ICube Laboratory (GAIA platform, CNRS / University of Strasbourg) in collaboration with LPC (Laboratory of Psychology Cognitive).

It aims to standardize and automate the scoring of the Hayling Sentence Completion Test, including its emotional variant, using modern NLP techniques and clinically interpretable rules.

---

## 🎥 Demo

🔗 [Demo video on Google Drive](https://drive.google.com/drive/folders/1wGJHunuULPBSd6BnFgGoU4K8hjbJ0zkj?usp=sharing)
---

## 🚀 Key Features

- ✅ Scoring pipeline using Word2Vec, FastText, LaBSE, and XLM-R-MNLI
- 📊 Threshold calibration using Balanced Accuracy
- 🧠 Rule-based system for proper noun and error categorization
- 🔎 Semantic similarity score computation
- 💬 Automatic GUI (PyQt5)
- 🧪 Evaluated on real and synthetic data from PsychoPy exports
- 📂 Balanced clinical labels, no data leakage, and no oversampling

---

## 🧱 Project Structure

📂 AI-hayling/
├── 📁 app/ → GUI and scoring logic
├── 📁 models/ → Pretrained models and fine-tuned embeddings
├── 📁 rules/ → Rule-based penalty assignment
├── 📁 scripts/ → Scripts for preprocessing, scoring, evaluation
├── 📁 labse_emotion/ → Fine-tuned LaBSE model (emotion version)
├── 📄 README.md → Project documentation
├── 📄 requirements.txt → Python dependencies
├── 📄 LICENSE → CeCILL v2.1 License


---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/insafhamdi/AI-hayling.git
cd AI-hayling
```
### 2. Create and activate virtual environment

#### On Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
#### On Windows (PowerShell):
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
## 📊 Usage

You can launch the graphical interface or run specific scoring/evaluation scripts.
To launch the GUI:
```bash
python app/gui_enhanced.py
```
To run batch evaluation on Hayling responses:
```bash
python scripts/evaluate_responses.py
```
## 📚 Citation & References
If you use this repo for research or clinical workflow pilots, cite:
```bash
Hamdi I., Lam J., Capobianco A., Tej I. (2025). An Automatic Scoring Method for responses to the Hayling Test. IEEE ICTAI 2025.
```
## 📄 License

📝 Licensed under the CeCILL v2.1 License — see LICENSE
 for more information.
For use cases in clinical or research settings only — no commercial deployment without approval.

<p align="right">
  <img src="https://raw.githubusercontent.com/insafhamdi/AI-hayling/main/assets/ictai.png" alt="ICTAI logo" width="200"/>
  <img src="https://raw.githubusercontent.com/insafhamdi/AI-hayling/main/assets/laboratoire-icube-logo-png_seeklogo-401018.png" alt="ICube logo" width="100"/>
</p> 

© 2025 – Insaf Hamdi, ICube (GAIA) & LPC
