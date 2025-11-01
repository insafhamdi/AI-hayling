# AI-hayling

# 🧠 EMOHayling – Automatic Scoring for the Emotional Hayling Test

[![License: CeCILL](https://img.shields.io/badge/license-CeCILL-blue.svg)](https://cecill.info)
[![Conference: ICTAI 2025](https://img.shields.io/badge/paper-ICTAI%202025-blueviolet)](https://www.ictai2025.org)

> A clinically-informed AI system to automatically score verbal inhibition in classical and emotional contexts — powered by NLP, embeddings, and fine-tuned transformers.

---

## 🎯 Project Overview

**AI-Hayling** is a research project developed during a 6-month internship at **ICube Laboratory (GAIA platform, CNRS / University of Strasbourg)** in collaboration with **LPC (Laboratoire de Psychologie Cognitive)**.

It aims to **standardize and automate the scoring** of the **Hayling Sentence Completion Test**, including its emotional variant, using modern NLP techniques and clinically interpretable rules.

---

## 🧪 Demo

https://drive.google.com/drive/folders/1wGJHunuULPBSd6BnFgGoU4K8hjbJ0zkj?usp=sharing


---

## 🧩 Key Features

- ✅ Scoring pipeline using **Word2Vec**, **FastText**, **LaBSE**, and **XLM-R-MNLI**
- 📊 Threshold optimization using **Balanced Accuracy**
- 🧠 Fine-tuned transformer models for better generalization
- 🤖 Synthetic data generation using **LLMs** (Phi-3, Zephyr)
- 🧍‍♀️ Interactive **PyQt5 GUI** for psychologists
- 📁 Automatic cleaning and scoring from **PsychoPy exports**
- 📈 Dashboard with inhibition rate, latency analysis, and item recommendations

---

## 🧱 Project Structure

AI-hayling/
│
├── app/ # Main application code (GUI, scoring logic)
├── EMOHayling_Interface/ # GUI files and assets
├── train_log.csv # Example training logs
├── requirements.txt # Python dependencies
├── README.md # You are here
└── LICENSE # CeCILL v2.1 license


---

## 🚀 Quick Start

Clone the repo:

on the bash: 
1) git clone https://github.com/insafhamdi/AI-hayling.git
2) cd AI-hayling

Create a virtual environment:


3) python -m venv .venv
.venv\Scripts\activate    # On Windows  
# OR  
source .venv/bin/activate  # On Linux/macOS

Install the dependencies:


4) pip install -r requirements.txt

Run the GUI:


5) python -m app.gui_enhanced


📚 Citation
If you use this work in your research or clinical workflow, please cite:

An Automatic Scoring Method for Responses to the Hayling Test.
I. Hamdi, J/ Lam-Weil, I. Abdeljaoued-Tej, E. Martz and A. Capobianco
In: IEEE International Conference on Tools with Artificial Intelligence (ICTAI 2025), Athens, Greece. [link soon]

📜 License
Distributed under the CeCILL v2.1 License – see LICENSE for more information.
For more info on CeCILL: https://cecill.info


 
