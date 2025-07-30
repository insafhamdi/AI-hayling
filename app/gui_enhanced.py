#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Hayling Émotionnel – Scorage automatique (LaBSE fine-tuned)
Distance sémantique uniquement + Active Learning
"""

from __future__ import annotations
import sys, json, re, unicodedata
from pathlib import Path
from typing import List

import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas

from PyQt5.QtCore    import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtGui     import QPixmap, QColor, QBrush
from PyQt5.QtWidgets import (
    QApplication, QTabWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QStatusBar, QTableView, QVBoxLayout, QWidget, QPushButton
)

# ------------------------------------------------------------------ chemins --
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
RES_DIR      = APP_DIR / "resources"

MODEL_DIR    = Path(r"C:\Users\hamdi\Desktop\ICube\EMOHayling\labse_emotion")
CLASSIF_CKPT = MODEL_DIR / "classifier.pt"
FEEDBACK_CSV = PROJECT_ROOT / "feedback_hayling.csv"

# ------------------------------------------------------------------ logos ----
def _logo(stem:str):
    for f in RES_DIR.iterdir():
        if f.is_file() and f.stem.lower().startswith(stem.lower()):
            return f
    return None
LOGOS={k:_logo(k) for k in ("lpc","icube","gaia")}

# ----------------------------------------------------------- chargement -----
def _first_notna(lst:List[pd.Series])->pd.Series:
    out=lst[0].copy()
    for s in lst[1:]:
        m=out.isna() & s.notna()
        out.loc[m]=s.loc[m]
    return out

def load_psychopy(p:Path)->pd.DataFrame:
    raw=pd.read_csv(p,sep=";",dtype=str)
    auto_mask  = raw.get("trials.textbox_Rep_4.text", pd.Series(dtype=str)).notna()
    inhib_mask = raw.get("trials_2.textbox_Rep_5.text", pd.Series(dtype=str)).notna()
    auto, inhib = raw.loc[auto_mask].copy(), raw.loc[inhib_mask].copy()
    auto ["Type de condition"]="Automatique"
    inhib["Type de condition"]="Inhibition"
    trials=pd.concat([auto,inhib]).sort_index()

    trials["Phrase à trou"]=_first_notna([
        trials.get("automatique_proposition_test",pd.Series(index=trials.index)),
        trials.get("inhibition_proposition_test", pd.Series(index=trials.index))]).str.strip()

    trials["Valence"]=_first_notna([
        trials.get("automatique_proposition_valence",pd.Series(index=trials.index)),
        trials.get("inhibition_proposition_valence", pd.Series(index=trials.index))]).str.strip().fillna("")

    trials["Réponse à inhiber / cible"]=_first_notna([
        trials.get("automatique_correct_proposition_test",pd.Series(index=trials.index)),
        trials.get("inhibition_correct_proposition_test", pd.Series(index=trials.index))]).str.lower().str.strip().fillna("")

    trials["Réponse du patient"]=_first_notna([
        trials.get("trials.textbox_Rep_4.text", pd.Series(index=trials.index)),
        trials.get("trials_2.textbox_Rep_5.text",pd.Series(index=trials.index))]).str.lower().str.strip().fillna("")

    trials["Temps (s)"]=_first_notna([
        trials.get("trials.key_resp_2.rt",pd.Series(index=trials.index)),
        trials.get("trials_2.key_rep.rt", pd.Series(index=trials.index))]).astype(float).round(3)

    cols=["Type de condition","Temps (s)","Phrase à trou","Valence",
          "Réponse à inhiber / cible","Réponse du patient"]
    return trials[cols].reset_index(drop=True)

# ------------------------ ROC → meilleur seuil ------------------------------
def opt_th(p,y):
    fpr,tpr,thr=roc_curve(y,p); youden=tpr-fpr
    return float(thr[np.argmax(youden)])

# ------------------------------------------------------- modèle Qt ----------
class PandasModel(QAbstractTableModel):
    COLOR={"Automatique":QBrush(QColor("#e8f5e9")),
           "Inhibition" :QBrush(QColor("#fff3e0"))}
    def __init__(self,df): super().__init__(); self._df=df
    def rowCount(self,*_): return len(self._df)
    def columnCount(self,*_): return len(self._df.columns)
    def data(self,idx,role=Qt.DisplayRole):
        if not idx.isValid(): return None
        v=self._df.iat[idx.row(),idx.column()]
        if role in (Qt.DisplayRole,Qt.EditRole):
            return "" if pd.isna(v) else str(v)
        if role==Qt.BackgroundRole:
            cond=self._df.iat[idx.row(),self._df.columns.get_loc("Type de condition")]
            return self.COLOR.get(cond)
    def headerData(self,i,ori,role=Qt.DisplayRole):
        return str(self._df.columns[i]) if role==Qt.DisplayRole and ori==Qt.Horizontal else super().headerData(i,ori,role)
    def flags(self,*_): return Qt.ItemIsSelectable|Qt.ItemIsEnabled|Qt.ItemIsEditable
    def setData(self,idx,val,role=Qt.EditRole):
        if role!=Qt.EditRole: return False
        try: self._df.iat[idx.row(),idx.column()]=int(val)
        except ValueError:   return False
        self.dataChanged.emit(idx,idx,[Qt.DisplayRole]); return True

# ---------------------------------------------------- fenêtre principale ----
class HaylingScorer(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Hayling Émotionnel – Scorage assisté"); self.resize(1400,800)
        self.labse=SentenceTransformer(str(MODEL_DIR),device="cpu")
        self.classif=torch.nn.Linear(768*3,2); self.classif.load_state_dict(torch.load(CLASSIF_CKPT,map_location="cpu")); self.classif.eval()

        self.sem_th=0.5; self._load_feedback()
        self._auto_scores=None

        # ----------------- UI root -------------------------------------------
        root=QVBoxLayout(); wrapper=QWidget(); wrapper.setLayout(root); self.setCentralWidget(wrapper)

        # Bannière logos
        ban=QHBoxLayout(); ban.addStretch()
        for p in LOGOS.values():
            if p:
                lbl=QLabel(); lbl.setPixmap(QPixmap(str(p)).scaledToHeight(70,Qt.SmoothTransformation)); ban.addWidget(lbl)
        ban.addStretch(); root.addLayout(ban)

        # Tabs
        self.tabs=QTabWidget(); root.addWidget(self.tabs)

        # ---- onglet Tableau
        tab_tbl=QWidget(); tbox=QVBoxLayout(tab_tbl)
        self.table=QTableView(); tbox.addWidget(self.table)
        btnL=QHBoxLayout()
        self.b_imp=QPushButton("Importer CSV"); self.b_imp.clicked.connect(self.import_csv); btnL.addWidget(self.b_imp)
        self.b_score=QPushButton("Coter"); self.b_score.setEnabled(False); self.b_score.clicked.connect(self.score); btnL.addWidget(self.b_score)
        self.b_save=QPushButton("Sauver corrections"); self.b_save.setEnabled(False); self.b_save.clicked.connect(self.save_feedback); btnL.addWidget(self.b_save)
        btnL.addStretch(); tbox.addLayout(btnL)
        self.tabs.addTab(tab_tbl,"Tableau")

        # ---- onglet Analyse
        tab_ana=QWidget(); anaL=QVBoxLayout(tab_ana)
        self.canvas=Canvas(plt.Figure(figsize=(8,4))); anaL.addWidget(self.canvas)
        self.lbl_stat=QLabel(); self.lbl_stat.setAlignment(Qt.AlignLeft|Qt.AlignTop); anaL.addWidget(self.lbl_stat)
        self.tabs.addTab(tab_ana,"Analyse"); self.tabs.setTabEnabled(1,False)

        self.status=QStatusBar(); self.setStatusBar(self.status)
        self.df=None
    
    def _refresh(self):
        self.table.setModel(PandasModel(self.df))
        self.table.resizeColumnsToContents()

    # ------------------ feedback --------------------------------------------
    def _load_feedback(self):
        if FEEDBACK_CSV.exists():
            fb=pd.read_csv(FEEDBACK_CSV)
            if {"p_sim","label"}.issubset(fb.columns):
                self.sem_th=opt_th(fb["p_sim"],(fb["label"]>0).astype(int))

    # ------------------ actions ---------------------------------------------
    def import_csv(self):
        fp,_=QFileDialog.getOpenFileName(self,"CSV PsychoPy",str(PROJECT_ROOT),"CSV (*.csv)")
        if not fp:return
        try: self.df=load_psychopy(Path(fp))
        except Exception as e: QMessageBox.critical(self,"Erreur import",str(e)); return
        self._refresh()
        self.b_score.setEnabled(True); self.b_save.setEnabled(False); self.tabs.setTabEnabled(1,False)
        self.status.showMessage("CSV chargé.")

    def score(self):
        if self.df is None: return

        norm = lambda t: re.sub(r"\s+"," ", unicodedata.normalize("NFKD", t)
            .encode("ascii","ignore").decode("ascii").lower()).strip()

        pats = [norm(x) for x in self.df["Réponse du patient"]]
        tgts = [norm(x) for x in self.df["Réponse à inhiber / cible"]]

        emb_p = self.labse.encode(pats, convert_to_tensor=True, normalize_embeddings=True)
        emb_t = self.labse.encode(tgts, convert_to_tensor=True, normalize_embeddings=True)

        cos_sim = torch.nn.functional.cosine_similarity(emb_p, emb_t, dim=1).cpu().numpy()
        dist_sem = 1 - np.clip(cos_sim, -1, 1)

        with torch.no_grad():
            feats = torch.cat([emb_p, emb_t, torch.abs(emb_p - emb_t)], dim=1)
            p_sim = self.classif(feats).softmax(-1)[:,1].cpu().numpy()

        scores = []
        for cond, p_txt, tgt_txt, ps in zip(self.df["Type de condition"], pats, tgts, p_sim):
            if cond == "Automatique":
                scores.append(0 if p_txt == tgt_txt else 1)
            else:
                scores.append(3 if p_txt == tgt_txt else (1 if ps >= self.sem_th else 0))

        # Ajout colonnes utiles
        self.df["Distance sémantique"]  = np.round(dist_sem, 4)
        self.df["p_sim"]                = np.round(p_sim, 4)
        self.df["Cotation automatique"] = scores
        self._auto_scores = scores.copy()

        self._refresh()
        self.b_save.setEnabled(True)
        self.tabs.setTabEnabled(1, True)
        self._refresh_analysis()
        self.status.showMessage("Cotation terminée.")

    def _refresh_analysis(self):
        fig = self.canvas.figure; fig.clf()

        # Camembert réussite/échec
        ax1 = fig.add_subplot(221)
        succ = (self.df["Cotation automatique"] == 0).sum()
        err  = len(self.df) - succ
        ax1.pie([succ, err],
            labels=[f"Correct (0) — {succ}", f"Erreur (1/3) — {err}"],
            autopct="%1.1f%%", startangle=90, colors=["#4C72B0", "#DD8452"])
        ax1.set_title("Réussite vs Échec")

        # Camembert par valence
        ax2 = fig.add_subplot(222)
        counts = self.df["Valence"].value_counts()
        ax2.pie(counts.values,
            labels=[f"{v} — {counts[v]}" for v in counts.index],
            autopct="%1.1f%%", startangle=90)
        ax2.set_title("Répartition des essais par Valence")

        # Temps moyen
        ax3 = fig.add_subplot(223); ax3.axis("off")
        temps_moy = self.df.groupby("Valence")["Temps (s)"].mean()
        slow = temps_moy.idxmax()
        txt = (f"Temps moyen par valence :\n"
               + "\n".join(f"• {v} : {temps_moy[v]:.1f}s" for v in temps_moy.index)
               + f"\n\n→ Plus lent pour « {slow} ».")
        ax3.text(0, 0.5, txt, va="center", fontsize=10)

        # Extrait métriques
        ax4 = fig.add_subplot(224); ax4.axis("off")
        extrait = self.df[["Distance sémantique", "Cotation automatique"]].head(5)
        ax4.text(0, 1, "Extrait des métriques (5 lignes) :", fontsize=10, va="top")
        ax4.text(0, 0.7, extrait.to_string(index=False), family="monospace", fontsize=8)

        fig.tight_layout(); self.canvas.draw()

    def save_feedback(self):
        mask = self.df["Cotation automatique"] != self._auto_scores
        if not mask.any():
            QMessageBox.information(self, "Aucune correction", "Aucune cotation n’a été modifiée.")
            return
        to_save = self.df.loc[mask, [
            "Type de condition","Phrase à trou","Valence",
            "Réponse à inhiber / cible","Réponse du patient",
            "Temps (s)","p_sim","Cotation automatique"
        ]].rename(columns={"Cotation automatique":"label"})
        header = not FEEDBACK_CSV.exists()
        to_save.to_csv(FEEDBACK_CSV, mode="a", header=header, index=False, encoding="utf-8")
        self.status.showMessage(f"{len(to_save)} corrections ajoutées dans {FEEDBACK_CSV.name}")
        QMessageBox.information(self, "Sauvegarde", f"{len(to_save)} corrections enregistrées.")                      

# ------------------------------------------------------------------ main -----
def main():
    app=QApplication(sys.argv); win=HaylingScorer(); win.show(); sys.exit(app.exec_())
if __name__=="__main__": main()
