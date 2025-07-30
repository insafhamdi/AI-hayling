import pandas as pd
import subprocess
import os

# Config
FICHIER_BASE = "data/responses_corrected.xlsx"
SEUIL_PATIENTS = 10  # seuil de déclenchement
COL_PATIENT = "patient_id"
COL_CORRECTED = "corrected_by_psy"

def verifier_et_lancer():
    if not os.path.exists(FICHIER_BASE):
        print("❌ Base non trouvée, arrêt.")
        return

    df = pd.read_excel(FICHIER_BASE)
    df_corrigees = df[df[COL_CORRECTED] == 1]
    n_patients = df_corrigees[COL_PATIENT].nunique()
    
    print(f"Patients corrigés : {n_patients}")
    
    if n_patients >= SEUIL_PATIENTS:
        print("✅ Seuil atteint, lancement de l’entraînement…")
        subprocess.run(["python", "scripts/active_learning.py"], check=True)
        print("✅ Entraînement terminé.")
    else:
        print("ℹ️ Seuil non atteint, rien à faire.")

if __name__ == "__main__":
    verifier_et_lancer()
