#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Active Learning pour Emotional Hayling
- Lit les feedbacks de la psychologue
- Détecte les nouvelles corrections
- Réentraine ou ajuste le modèle
- Sauvegarde le modèle et exporte les cas incertains
"""

import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# --------------------
# CONFIG
# --------------------
FEEDBACK_PATH = "feedback_hayling.csv"
MODEL_DIR = "model"
LOG_DIR = "logs"
UNCERTAIN_EXPORT = "data/uncertain_cases.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Modèle initial (ex : logistic regression sur embeddings LaBSE)
MODEL_PATH = os.path.join(MODEL_DIR, "active_model.joblib")

# --------------------
# 1. Chargement feedback
# --------------------
print("[INFO] Chargement du fichier de feedback...")
df = pd.read_csv(FEEDBACK_PATH)

# Supposons que df contient au moins :
# - reponse_a_inhiber
# - reponse
# - penalite_humaine (0 ou 1)
# - embedding_* (colonne(s) vecteurs LaBSE)
embedding_cols = [col for col in df.columns if col.startswith("emb_")]

if len(embedding_cols) == 0:
    raise ValueError("Aucune colonne d'embeddings trouvée (nommées emb_*)")

# Features et labels
X = df[embedding_cols].values
y = df["penalite_humaine"].values

# --------------------
# 2. Charger ou créer modèle
# --------------------
if os.path.exists(MODEL_PATH):
    print("[INFO] Chargement du modèle existant...")
    model = joblib.load(MODEL_PATH)
else:
    print("[INFO] Création d'un nouveau modèle...")
    model = LogisticRegression(max_iter=1000)

# --------------------
# 3. Réentraînement / mise à jour
# --------------------
print("[INFO] Entraînement du modèle avec feedback...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"[RESULT] Accuracy: {acc:.3f} | F1: {f1:.3f}")

# Sauvegarde modèle
joblib.dump(model, MODEL_PATH)
print(f"[INFO] Modèle sauvegardé dans {MODEL_PATH}")

# --------------------
# 4. Détection des cas incertains
# --------------------
proba = model.predict_proba(X)
uncertain_mask = (proba.max(axis=1) < 0.6)  # seuil d'incertitude
uncertain_cases = df.loc[uncertain_mask]

if len(uncertain_cases) > 0:
    uncertain_cases.to_csv(UNCERTAIN_EXPORT, index=False)
    print(f"[INFO] {len(uncertain_cases)} cas incertains exportés vers {UNCERTAIN_EXPORT}")

# --------------------
# 5. Logging
# --------------------
log_file = os.path.join(LOG_DIR, f"active_learning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
with open(log_file, "w") as f:
    f.write(f"Accuracy: {acc:.3f}\nF1: {f1:.3f}\n")
    f.write(f"Cas incertains: {len(uncertain_cases)}\n")

print(f"[INFO] Log sauvegardé : {log_file}")
print("[INFO] Active Learning terminé ✅")
