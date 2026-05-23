# 🌾 Prédiction du Rendement Agricole avec RAPIDS cuML + SHAP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/votre-username/agriculture-yield-gpu/blob/main/agriculture_yield_pipeline.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-cuML-green.svg)](https://rapids.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

Pipeline ML complet pour prédire le **rendement des cultures agricoles** (hg/ha) à partir des données historiques de la FAO (FAOSTAT). Le pipeline exploite l'accélération GPU via **RAPIDS cuML** pour entraîner trois modèles :

- 🌲 **Random Forest** (cuML)
- ⚡ **XGBoost** (GPU)
- 📈 **Régression Linéaire** (cuML)

L'explicabilité est assurée par **SHAP** (SHapley Additive exPlanations) avec visualisations interactives.

---

## 🎯 Objectifs

| Objectif | Détail |
|----------|--------|
| **Dataset** | FAO FAOSTAT – 28 000+ lignes (pays × cultures × années) |
| **Cible** | Rendement en hg/ha (`yield_hg_ha`) |
| **Métriques** | RMSE, R² (train & test) |
| **GPU** | NVIDIA T4 / K80 (Google Colab gratuit) |
| **Explicabilité** | SHAP summary plots (bar + dot) |

---

## 🚀 Instructions d'exécution sur Google Colab

### Étape 1 – Ouvrir le notebook
1. Cliquer sur le badge **Open in Colab** ci-dessus, **OU**
2. Aller sur [colab.research.google.com](https://colab.research.google.com) → `Fichier` → `Ouvrir un notebook` → `GitHub` → coller l'URL du repo.

### Étape 2 – Activer le GPU
> **IMPORTANT** : Sans GPU, RAPIDS ne peut pas s'installer.

`Modifier` → `Paramètres du notebook` → **Accélérateur matériel** → **GPU (T4 recommandé)**

### Étape 3 – Lancer toutes les cellules
`Exécution` → `Tout exécuter` (ou `Ctrl+F9`)

> ⏱️ **Durée estimée** : ~15–25 min (dont ~10 min pour l'installation RAPIDS)

---

## 📦 Prérequis

### Environnement
- Google Colab avec GPU activé (T4 ou K80)
- Python 3.10+ (fourni par Colab)
- Connexion Internet (pour télécharger RAPIDS et le dataset)

### Bibliothèques installées automatiquement
| Bibliothèque | Version | Usage |
|---|---|---|
| `cudf` | ≥24.x | DataFrames GPU (RAPIDS) |
| `cuml` | ≥24.x | ML sur GPU (RAPIDS) |
| `xgboost` | ≥2.0 | Gradient Boosting GPU |
| `shap` | ≥0.45 | Explicabilité SHAP |
| `matplotlib` | ≥3.7 | Visualisations |
| `seaborn` | ≥0.12 | Heatmaps & styles |
| `pandas` | ≥2.0 | Manipulation données CPU |
| `scikit-learn` | ≥1.3 | Fallback CPU + métriques |
| `numpy` | ≥1.24 | Calculs numériques |

> ℹ️ Sur Colab, RAPIDS s'installe via la méthode officielle `rapidsai/rapidsai-csp` qui prend en charge la version CUDA automatiquement.

---

## 📊 Résultats Attendus

### Performances typiques (dataset FAO réel)

| Modèle | RMSE (test) | R² (test) |
|--------|-------------|-----------|
| Random Forest (cuML) | ~15 000 | ~0.88 |
| XGBoost (GPU) | ~13 500 | ~0.91 |
| Régression Linéaire (cuML) | ~28 000 | ~0.65 |

> Les performances varient selon le dataset chargé et les hyperparamètres.

### Graphiques générés
1. **Heatmap des corrélations** – Relations entre features numériques
2. **Distribution du rendement** – Histogramme avec KDE
3. **Comparaison des modèles** – Barplot RMSE & R²
4. **SHAP Summary Plot (bar)** – Importance globale des features
5. **SHAP Summary Plot (dot)** – Impact directionnel des features
6. **SHAP Waterfall** – Explication d'une prédiction individuelle

---

## 🗂️ Structure du Projet

```
agriculture-yield-gpu/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances (référence)
├── environment.yml                    # Conda environment (référence)
├── .gitignore                         # Fichiers à ignorer
├── agriculture_yield_pipeline.ipynb   # Notebook principal
└── setup_colab.sh                     # Script d'installation (optionnel)
```

---

## 🔧 Fonctionnalités Techniques

### Robustesse
- ✅ **Fallback automatique** : Si RAPIDS n'est pas disponible → scikit-learn CPU avec warning
- ✅ **Dataset synthétique** : Si l'URL FAO échoue → génération d'un dataset réaliste (28 000 lignes)
- ✅ **Gestion NaN** : Imputation médiane pour les numériques, mode pour les catégorielles
- ✅ **Blocs try/except** : Toutes les étapes critiques sont protégées

### Pipeline ML
- Encodage des variables catégorielles (LabelEncoder)
- Normalisation StandardScaler (pour la régression linéaire)
- Split stratifié train/test (80/20)
- Conversion float32 pour cuML
- Validation croisée des hyperparamètres

### SHAP
- TreeExplainer pour Random Forest et XGBoost
- LinearExplainer pour la Régression Linéaire
- Summary plots sauvegardés en PNG

---

## ⚠️ Notes Importantes

1. **Redémarrage Colab** : L'installation RAPIDS peut nécessiter un redémarrage du kernel. Le notebook le détecte et vous guide.
2. **Quota GPU Colab** : Le GPU gratuit est limité (~12h/jour). Si la session expire, relancez.
3. **Version CUDA** : Le script détecte automatiquement la version CUDA de votre instance.

---

## 📚 Références

- [RAPIDS Documentation](https://docs.rapids.ai/)
- [FAO FAOSTAT](https://www.fao.org/faostat/en/#data)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [cuML User Guide](https://docs.rapids.ai/api/cuml/stable/)

---

## 📝 Licence

MIT License – voir [LICENSE](LICENSE) pour les détails.

---

*Projet réalisé dans le cadre d'une démonstration ML agricole avec accélération GPU.*
