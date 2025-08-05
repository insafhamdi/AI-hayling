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

import datetime
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_curve, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas

from PyQt5.QtCore    import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtGui     import QPixmap, QColor, QBrush
from PyQt5.QtWidgets import (
    QApplication, QTabWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QStatusBar, QTableView, QVBoxLayout, QWidget, QPushButton,
    QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout
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
    pid_col = next((c for c in raw.columns if c.lower().startswith(("n° participant", "dp"))), None)
    if pid_col:
        pid_val = raw[pid_col].dropna().iloc[0]
    else:
        pid_val = "Inconnu"    
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
    df = trials[cols].reset_index(drop=True)
    df.attrs["patient_id"] = str(pid_val)
    return df

# ------------------------ ROC → meilleur seuil ------------------------------
def opt_th(p,y):
    fpr,tpr,thr=roc_curve(y,p); youden=tpr-fpr
    return float(thr[np.argmax(youden)])

# ------------------------------------------------------- modèle Qt ----------
class PandasModel(QAbstractTableModel):
    COLOR={"Automatique":QBrush(QColor("#e8f5e9")),
        "Inhibition" :QBrush(QColor("#fff3e0"))}
    RED = QBrush(QColor("#FF8A80"))
    def __init__(self,df): super().__init__(); self._df=df
    def rowCount(self,*_): return len(self._df)
    def columnCount(self,*_): return len(self._df.columns)
    def data(self,idx,role=Qt.DisplayRole):
        if not idx.isValid(): return None
        v=self._df.iat[idx.row(),idx.column()]
        if role in (Qt.DisplayRole,Qt.EditRole):
            return "" if pd.isna(v) else str(v)
        if role==Qt.BackgroundRole:
            # met en rouge si la correction humaine a eu lieu 
            coll_corr = "corrigé"
            if coll_corr in self._df.columns and self._df.iloc[idx.row()][coll_corr]:
                return self.RED
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
        super().__init__(); self.setWindowTitle("EMOHayling – Cotation assisté"); self.setWindowState(Qt.WindowMaximized)

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
                lbl=QLabel(); lbl.setPixmap(QPixmap(str(p)).scaledToHeight(100,Qt.SmoothTransformation)); ban.addWidget(lbl)
        ban.addStretch(); root.addLayout(ban)

        # ID patient (à afficher en haut à droite)
        self.lbl_patient = QLabel()
        self.lbl_patient.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.lbl_patient.setTextFormat(Qt.RichText)
        self.lbl_patient.setText(f"""
        <span style='
            color:#1976D2;
            font-size:14pt;
            font-family:"Seoge UI","Arial Rounded MT Bold", Arial, sans-serif;
            font-weight:600;
            letter-spacing:1px;
            background: #F5FAFF;
            border-radius: 8px;
            padding: 2px 18px 2px 12 px;
        '>
        ID participant 
        </span>
        """)
        root.addWidget(self.lbl_patient)
        if FEEDBACK_CSV.exists():
            n_feedback = len(pd.read_csv(FEEDBACK_CSV))
        else:
            n_feedback = 0
        self.lbl_feedback= QLabel()
        self.lbl_feedback.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.lbl_feedback.setTextFormat(Qt.RichText)
        self.lbl_feedback.setStyleSheet("color:#1565c0; font-size:10.5pt; padding-right:18px;margin-top: -7px;")
        txt = f"<i>{n_feedback} feedbacks disponibles pour l'apprentissage</i>" if n_feedback > 0 else ""
        self.lbl_feedback.setText(txt)
        root.addWidget(self.lbl_feedback)
        
        

        # Tabs
        self.tabs=QTabWidget(); root.addWidget(self.tabs)

        # ---- onglet Tableau
        tab_tbl=QWidget(); tbox=QVBoxLayout(tab_tbl)
        self.table=QTableView(); tbox.addWidget(self.table)
        btnL=QHBoxLayout()
        self.b_imp=QPushButton("Importer CSV"); self.b_imp.clicked.connect(self.import_csv); btnL.addWidget(self.b_imp)
        self.b_score=QPushButton("Coter"); self.b_score.setEnabled(False); self.b_score.clicked.connect(self.score); btnL.addWidget(self.b_score)
        self.b_save=QPushButton("Sauver corrections"); self.b_save.setEnabled(False); self.b_save.clicked.connect(self.save_feedback); btnL.addWidget(self.b_save)
        self.b_preview_reco = QPushButton("Prévisualiser recommandations")
        
        self.b_preview_reco.clicked.connect(self.preview_recommandations)
        btnL.addWidget(self.b_preview_reco)
        # ajouter le bouton dee reéntrainement 
        self.b_retrain = QPushButton("Réentraîner le modèle")
        self.b_retrain.setEnabled(True)
        self.b_retrain.clicked.connect(self.retrain_model)
        btnL.addWidget(self.b_retrain)
        
        btnL.addStretch(); tbox.addLayout(btnL)
        self.tabs.addTab(tab_tbl,"Tableau")

        # ---- onglet Analyse
        tab_ana=QWidget(); anaL=QVBoxLayout(tab_ana)
        lbl_dash = QLabel("""<span style='color:#1565c0;font-size:24px;font-family:"Arial Black",Arial, sans-serif;letter-spacing:1px; font-weight:900;'>Tableau de bord EMOHayling – Analyse</span>""")
        lbl_dash.setAlignment(Qt.AlignCenter)
        lbl_dash.setTextFormat(Qt.RichText)
        anaL.addWidget(lbl_dash)
        
        
        
        self.canvas=Canvas(plt.Figure(figsize=(13,8))); anaL.addWidget(self.canvas, stretch=1)
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
        try:
            self.df = load_psychopy(Path(fp))
            pid = self.df.attrs.get("patient_id", "inconnu")
            self.lbl_patient.setText(f"""
            <span style='
                color:#1976D2;
                font-size:14pt;
                font-family:"Segoe UI","Arial Rounded MT Bold", Arial, sans-serif;
                font-weight:600;
                letter-spacing:1px;
                background: #F5FAFF;
                border-radius: 8px;
                padding: 2px 18px 2px 12px;'>
            ID participant : {pid}
            </span>
            """)
        except Exception as e:
            QMessageBox.critical(self, "Erreur import", str(e))
            return
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
        fig = self.canvas.figure
        fig.clf()
        fig.set_size_inches(16, 5)  # large et bas
        gs = fig.add_gridspec(1,4,wspace=0.45) 

        df = self.df.copy()

        # -------- Bloc 1 : Taux de réussite/échec ---------
        ax1 = fig.add_subplot(gs[0,0])
        succ = (df["Cotation automatique"] == 0).sum()
        err  = (df["Cotation automatique"] != 0).sum()
        # Bleu et vert pastel
        pastel_colors = ["#90caf9", "#a5d6a7"]
        wedges, texts, autotexts = ax1.pie(
            [succ, err],
            labels=[f"Réussite ({succ})", f"Erreur ({err})"],
            autopct="%1.1f%%",
            startangle=90,
            colors=pastel_colors,
            textprops={'fontsize': 11, 'color': '#233'}
        )
        # Agrandir le texte au centre
        for autotext in autotexts:
            autotext.set_fontsize(12)
        ax1.set_title("Réussite vs Échec", fontsize=13, pad=10)
        ax1.axis('equal')

        # -------- Bloc 2 : Temps moyen par valence ---------
        ax2 = fig.add_subplot(gs[0,1])
        temps_moy = df.groupby("Valence")["Temps (s)"].mean().sort_values()
        pastel_bar = ["#90caf9", "#a5d6a7", "#b2dfdb"][:len(temps_moy)]
        if not temps_moy.empty:
            bars = ax2.bar(temps_moy.index, temps_moy.values, color=pastel_bar)
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f"{height:.1f}s",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11)
            ax2.set_ylim(0, max(temps_moy.max()*1.1, 2))
        ax2.set_title("Temps moyen par valence", fontsize=13, pad=10)
        ax2.set_ylabel("Temps moyen (s)")
        ax2.tick_params(axis='x', labelsize=11, rotation=10)

        # -------- Bloc 3 : Items à surveiller (si multi-patients) ---------
        ax3 = fig.add_subplot(gs[0,2])
        fail_rate = (df.groupby("Phrase à trou")["Cotation automatique"]
                    .apply(lambda x: (x == 3).mean()))
        surveil_items = fail_rate[fail_rate >= 0.4].sort_values(ascending=False).head(5)
        ax3.axis("off")
        if not surveil_items.empty:
            msg = "★ Items à surveiller (>40% échec) ★\n\n"
            for item, rate in surveil_items.items():
                msg += f"• {item[:45]}... : {rate*100:.0f}%\n"
            ax3.text(0, 1, msg, fontsize=11, va="top", family="monospace", color="#00695c")
        else:
            ax3.text(0.5, 0.5, "Aucun item à surveiller", ha="center", va="center", fontsize=12, color="#888")
        self._plot_evolution_curve(fig, gs[0,3])
        fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.18)
        self.canvas.draw()
        
        # -------- Ajustement global ---------
        fig.subplots_adjust(left=0.05, right=0.97, wspace=0.35, top=0.86, bottom=0.16)

        # Pas de titre dans le canvas (plus propre)
        self.canvas.draw()
    def _plot_evolution_curve(self, fig,position):
        # ajoute une courbe d'evolution du taux d'echec/erreur cumulé (feedback_hayling.csv)
        if not FEEDBACK_CSV.exists():
            return
        fb = pd.read_csv(FEEDBACK_CSV)
        if len(fb) < 2 or "label" not in fb.columns:
            return
        is_error = fb["label"].apply(lambda x: int(x==3))
        n = np.arrange(1, len(is_error)+1)
        taux_cumule = is_error.cumsum() / n  
        
        ax = fig.add_subplot(position)
        ax.plot(n, taux_cumule*100, color="#e57373", lw=2, label="Taux d'échec cumulé")
        ax.set_title("Evolution du taux d'échec (score=3)", fontsize=12)
        ax.set_xlabel("Nombre de feedbacks cumulés")
        ax.set_ylabel("Taux d'échex (%)")
        ax.set_ylim(0,100)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=10)
    def preview_recommandations(self):
        if not FEEDBACK_CSV.exists():
            QMessageBox.warnings(self,"Aucun feedback","Aucun fichier de feedback trouvé pour générer les recommandations.")
            return 
        df = pd.read_csv(FEEDBACK_CSV)
        # on garde uniquement les items "Inhibition" pour la standardisation 
        df = df[df["Type de condition"] == "Inhibition"]
        
        # statistiques par item & valence 
        stats =(df.groupby(["Valence", "Phrase à trou"])
                .agg(
                    n=("label", "size"),
                    n_reussite=("label",lambda x: (x==0).sum()),
                    n_ambigu=("label", lambda x: (x == 1).sum()),
                    n_echec = ("label", lambda x: (x==3).sum()),
                    temps_moy=("Temps (s)", "mean")
                    
        
                )
                .reset_index()
                )
        stats["%/ réussite"] = stats["n_reussite"] / stats["n"]
        stats["%/ echec"] = stats["n_echec"] / stats["n"]
        stats["%/ ambigu"] = stats["n_ambigu"] / stats["n"]
        
        # classification automatique
        def cat_reco(row):
            if row["n"] < 5:
                return "Données insuffisantes"
            if row["%/ réussite"] > 0.9:
                return "Trop facile"
            if row["%/ echec"] > 0.6:
                return "Trop difficile"
            if row["%/ ambigu"] > 0.3:
                return "Ambigu"
            if 0.2 <= row["%/ echec"] <= 0.5:
                return " A recommander"
            return "OK"
        stats["Catégorie reco"] = stats.apply(cat_reco, axis=1)
        # trie pour que la psy voit d'abord 'a recommander'
        stats = stats.sort_values(["Valence","Catégorie reco","%/ echec"], ascending=[True, True, False])
        # garder les colonnes utiles 
        export_cols = ["Valence", "Phrase à trou","n","%/ réussite", "%/ echec","%/ ambigu","temps_moy","Catégorie reco"]
        stats_export = stats[export_cols].copy()
        stats_export = stats_export.round(3)
        
        # affichage dans une fenetre de preview Qt 
        dialog = QDialog(self)
        dialog.setWindowTitle("Aperçu des recommandations d'items")
        layout = QVBoxLayout(dialog)
        table = QTableWidget(dialog)
        table.setColumnCount (len(export_cols))
        table.setHorizontalHeaderLabels(export_cols)
        table.setRowCount(len(stats_export))
        for row in range(len(stats_export)):
            for col, colname in enumerate(export_cols):
                val = stats_export.iloc[row, col]
                item = QTableWidgetItem(str(val))
                
                # couleur selon recommandation
                if colname == "Catégorie recommandation":
                    if val == "A recommander":
                        item.setBackground(QColor("#a5d6a7")) # vert doux
                    elif val == "Trop difficile":
                        item.setBackground(QColor("#ffe082")) # jaune doux
                    elif val == "Trop facile":
                        item.setBackground(QColor("#b3e5fc")) # bleu doux
                    elif val == "Ambigu":
                        item.setBackground(QColor("#ce93d8")) # violet clair
                    elif val == "Données insuffisantes":
                        item.setBackground(QColor("#e0e0e0")) # gris 
                table.setItem(row, col, item)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        # bouton export dans la fenetre de preview
        def do_export():
            dt = datetime.datetime.now().strftime("%Y%nm%d_%H%M%S")
            out_csv = PROJECT_ROOT / f"items_recomandes_{dt}.csv"
            stats_export.to_csv(out_csv, index=False, encoding="utf-8-sig")
            QMessageBox.information(dialog, "Export terminé", f"Recommandations exportées vers {out_csv.name}")
            
        btn_export = QPushButton("Exporter au format CSV")
        btn_export.clicked.connect(do_export)
        layout.addWidget(btn_export)
        dialog.setLayout(layout)
        dialog.resize(1100,600)
        dialog.exec_()
        
    

        
        
        


    def save_feedback(self):
        # 1. Ajoute la colonne "corrigé" : True si modifiée par la psy, False sinon
        corrige = (self.df["Cotation automatique"] != self._auto_scores)
        df_to_save = self.df.copy()
        df_to_save["corrigé"] = corrige
        df_to_save = df_to_save.rename(columns={"Cotation automatique":"label"})
        df_to_save["source_label"] = np.where(corrige, "psy", "auto")
        

        # 2. (Optionnel) Ajoute l'identifiant patient si dispo dans self.df
        # Si tu veux le gérer : il faut une colonne 'patient_id' dans le dataframe à la base
        # Si pas présent, retire cette ligne ou adapte selon tes colonnes dispo

        # 3. Fusion intelligente dans le CSV
        header = not FEEDBACK_CSV.exists()
        if FEEDBACK_CSV.exists():
            old = pd.read_csv(FEEDBACK_CSV)
            # Fusion/dédoublonnage par colonnes clés
            keys = ["Phrase à trou", "Réponse du patient"]
            if "patient_id" in df_to_save.columns:
                keys.append("patient_id")
            all_ = pd.concat([old, df_to_save], ignore_index=True)
            all_ = all_.drop_duplicates(subset=keys, keep="last")
            all_.to_csv(FEEDBACK_CSV, index=False, encoding="utf-8")
            n_sauve = len(df_to_save)
        else:
            df_to_save.to_csv(FEEDBACK_CSV, index=False, header=header, encoding="utf-8")
            n_sauve = len(df_to_save)

        self.status.showMessage(f"{n_sauve} réponses ajoutées/actualisées dans {FEEDBACK_CSV.name}")
        QMessageBox.information(self, "Sauvegarde", f"{n_sauve} réponses enregistrées (dont corrections).")
        
    
    def retrain_model(self):
        if not FEEDBACK_CSV.exists():
            return
        df = pd.read_csv(FEEDBACK_CSV)
        if len(df) == 0:
            self.status.showMessage("Feedback vide, rien à entrainer.")
            return

        # Feature engineering identique à la cotation
        norm = lambda t: re.sub(r"\s+"," ", unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode("ascii").lower()).strip()
        pats = [norm(x) for x in df["Réponse du patient"].astype(str)]
        tgts = [norm(x) for x in df["Réponse à inhiber / cible"].astype(str)]
        emb_p = self.labse.encode(pats, convert_to_tensor=True, normalize_embeddings=True)
        emb_t = self.labse.encode(tgts, convert_to_tensor=True, normalize_embeddings=True)
        feats = torch.cat([emb_p, emb_t, torch.abs(emb_p - emb_t)], dim=1)
        y = torch.tensor(df["label"].values, dtype=torch.long)

        # Split train/val (stratifié si possible)
        X_train, X_val, y_train, y_val = train_test_split(
            feats, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        # Réinitialise le classifieur
        classif = torch.nn.Linear(feats.shape[1], 2)
        opt = torch.optim.Adam(classif.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_bacc = 0
        best_weights = None
        patience = 5
        wait = 0
        min_val_loss = float('inf')

        # Entraînement avec early stopping et évaluation
        for epoch in range(30):  # Max 30 epochs
            classif.train()
            opt.zero_grad()
            output = classif(X_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            opt.step()

            # Validation
            classif.eval()
            with torch.no_grad():
                val_out = classif(X_val)
                val_loss = loss_fn(val_out, y_val).item()
                preds = torch.argmax(val_out, dim=1)
                y_val_np = y_val.cpu().numpy() if hasattr(y_val, "cpu") else y_val.numpy()
                preds_np = preds.cpu().numpy() if hasattr(preds, "cpu") else preds.numpy()
                bacc = balanced_accuracy_score(y_val_np, preds_np)
                

            # Early stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_bacc = bacc
                best_weights = classif.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        # Charge les meilleurs poids
        if best_weights is not None:
            classif.load_state_dict(best_weights)

        torch.save(classif.state_dict(), CLASSIF_CKPT)
        self.classif.load_state_dict(torch.load(CLASSIF_CKPT, map_location="cpu"))
        self.classif.eval()

        # Journalisation log
        log_file = PROJECT_ROOT / "train_log.csv"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "datetime": now,
            "n_feedback": len(df),
            "val_bacc": best_bacc,
            "val_loss": min_val_loss,
            "epochs": epoch+1
        }
        # Append log
        if not log_file.exists():
            pd.DataFrame([log_data]).to_csv(log_file, index=False)
        else:
            pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=False, index=False)

        # Affichage feedback
        msg = (f"Réentraînement terminé – Balanced accuracy validation: {best_bacc:.2%} "
            f"(n={len(y_val)}), epochs: {epoch+1}")
        self.status.showMessage(msg)
        QMessageBox.information(self, "Réentraînement modèle", msg)
        
# ------------------------------------------------------------------ main -----
def main():
    app=QApplication(sys.argv); win=HaylingScorer(); win.show(); sys.exit(app.exec_())
if __name__=="__main__": main()
