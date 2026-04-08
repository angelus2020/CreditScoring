import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score,
    classification_report, confusion_matrix,
    precision_recall_curve, auc
)


class Evaluation:
    def __init__(
        self,
        model,
        x: np.ndarray,
        y: np.ndarray,
        features: list = None,
        figure_dir: str = "../../figuras/s12/"
    ):
        self.model      = model
        self.x          = x
        self.y          = y
        self.features   = features
        self.figure_dir = figure_dir
        self.cut_off    = None          # se calcula en evaluate()

    # ------------------------------------------------------------------ #
    def evaluate(self) -> tuple[np.ndarray, np.ndarray]:
        y_proba = self.model.predict_proba(self.x)[:, 1]

        fpr, tpr, thresholds = roc_curve(self.y, y_proba)
        ks_values  = tpr - fpr
        best_idx   = np.argmax(ks_values)
        self.cut_off = float(thresholds[best_idx])

        y_pred = (y_proba >= self.cut_off).astype(int)

        print(f"\n── Test Set Metrics (OOT)  —  threshold = {self.cut_off:.4f} ────────")
        print(f"  ROC AUC          : {roc_auc_score(self.y, y_proba):.4f}")
        print(f"  PR AUC           : {average_precision_score(self.y, y_proba):.4f}")
        print(f"  Recall           : {recall_score(self.y, y_pred):.4f}")
        print(f"  Precision        : {precision_score(self.y, y_pred):.4f}")
        print(f"  F1 Score         : {f1_score(self.y, y_pred):.4f}")
        print(f"\n{classification_report(self.y, y_pred, target_names=['Good (0)', 'Bad (1)'])}")

        return y_proba, y_pred

    # ------------------------------------------------------------------ #
    def plot_evaluation(self, y_proba: np.ndarray, y_pred: np.ndarray) -> None:
        precisions, recalls, _ = precision_recall_curve(self.y, y_proba)
        pr_auc = auc(recalls, precisions)
        fpr, tpr, _ = roc_curve(self.y, y_proba)
        cm = confusion_matrix(self.y, y_pred)   # filas = Actual, cols = Predicted

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Model — OOT Evaluation', fontsize=13, fontweight='bold')

        # Confusion Matrix
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Good', 'Bad'],
            yticklabels=['Good', 'Bad']
        )
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        # ROC Curve
        axes[1].plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}', color='steelblue')
        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        axes[1].set_title('ROC Curve')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend()

        # Precision-Recall Curve
        axes[2].plot(recalls, precisions, label=f'PR AUC = {pr_auc:.3f}', color='darkorange')
        baseline = float(self.y.mean())

        axes[2].axhline(
            baseline, color='grey', linestyle='--',
            linewidth=0.8, label=f'Baseline = {baseline:.2%}'
        )
        axes[2].set_title('Precision-Recall Curve')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].legend()

        plt.tight_layout()
        os.makedirs(self.figure_dir, exist_ok=True)
        fig_path = os.path.join(self.figure_dir, 'evaluation.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"[plot]  Saved → '{fig_path}'")

    # ------------------------------------------------------------------ #
    def plot_feature_importance(self) -> None:
        """
        Bar chart of feature importances homologated to GAIN
        (XGBoost gain, LightGBM gain, sklearn MDI ~ gain-like)
        """

        # --- Pipeline or direct model ---
        if hasattr(self.model, 'named_steps'):
            model_step = self.model.named_steps['model']
        else:
            model_step = self.model

        model_type = type(model_step).__name__
        importances = None
        importance_label = 'Gain'

        # -------------------------------------------------
        # sklearn tree-based (approximation to gain)
        # -------------------------------------------------
        if hasattr(model_step, 'feature_importances_') and not hasattr(model_step, 'get_booster'):
            importances = model_step.feature_importances_
            importance_label = 'Gain-like (Mean Decrease Impurity)'

        # -------------------------------------------------
        # XGBoost (real gain)
        # -------------------------------------------------
        elif hasattr(model_step, 'get_booster'):
            try:
                booster = model_step.get_booster()
                scores = booster.get_score(importance_type='gain')

                importances = (
                    pd.Series(scores)
                    .reindex(self.features)
                    .fillna(0)
                    .values
                )
                importance_label = 'Gain (XGBoost)'
            except Exception as e:
                print(f"[plot_feature_importance] XGBoost error: {e}")
                return

        # -------------------------------------------------
        # LightGBM (real gain)
        # -------------------------------------------------
        elif hasattr(model_step, 'booster_'):
            try:
                importances = model_step.booster_.feature_importance(
                    importance_type='gain'
                )
                importance_label = 'Gain (LightGBM)'
            except Exception as e:
                print(f"[plot_feature_importance] LightGBM error: {e}")
                return

        else:
            print(f"[plot_feature_importance] Model '{model_type}' not supported.")
            return

        # --- Optional normalization (recommended for plotting) ---
        if importances.sum() > 0:
            importances = importances / importances.sum()

        importance_label += ' (normalized)'

        # --- Plot ---
        importances_series = (
            pd.Series(importances, index=self.features)
            .sort_values(ascending=True)
        )

        fig, ax = plt.subplots(figsize=(8, max(4, len(importances_series) * 0.35)))
        importances_series.plot(kind='barh', ax=ax, color='#2ca02c')

        ax.set_title(
            f'Feature Importance — {model_type}\n(All models shown as GAIN)',
            fontsize=11
        )
        ax.set_xlabel(importance_label)

        plt.tight_layout()

        os.makedirs(self.figure_dir, exist_ok=True)
        fig_path = os.path.join(self.figure_dir, 'feature_importance.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"[plot] Saved → {fig_path}")

    # ------------------------------------------------------------------ #
    def run_all(self) -> None:
        y_proba, y_pred = self.evaluate()
        self.plot_evaluation(y_proba, y_pred)
        self.plot_feature_importance()
        print("\n✅  Evaluation completed.")