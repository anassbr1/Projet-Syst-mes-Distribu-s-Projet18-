# 🌾 Prédiction du Rendement Agricole avec RAPIDS cuML + SHAP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/votre-username/agriculture-yield-gpu/blob/main/agriculture_yield_pipeline.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-cuML-green.svg)](https://rapids.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

Pipeline ML complet pour prédire le **rendement des cultures agricoles** (hg/ha) à partir des **données historiques de la FAO** (FAOSTAT). Le pipeline exploite l’accélération GPU via **RAPIDS cuML** pour entraîner trois modèles :

- 🌲 **Random Forest** (cuML / scikit‑learn fallback)
- ⚡ **XGBoost** (avec support GPU `device='cuda'`)
- 📈 **Régression Linéaire** (cuML / scikit‑learn fallback)

L’explicabilité est assurée par **SHAP** (SHapley Additive exPlanations) avec visualisations interactives (summary plots bar + dot, waterfall plot).

---

## 🎯 Objectifs

| Objectif | Détail |
|----------|--------|
| **Dataset** | FAOSTAT (Kaggle : `vijayveersingh/faostat-crops-and-livestock-data`) – 28 000+ lignes |
| **Cible** | Rendement en hg/ha (`yield_hg_ha`) |
| **Métriques** | RMSE, R² (train & test) |
| **GPU** | NVIDIA T4 / K80 (Google Colab gratuit) |
| **Explicabilité** | SHAP summary plots (bar + dot) + waterfall |

---

## 🚀 Instructions d’exécution sur Google Colab

### Étape 1 – Ouvrir le notebook
1. Cliquer sur le badge **Open in Colab** ci-dessus, **OU**
2. Aller sur [colab.research.google.com](https://colab.research.google.com) → `Fichier` → `Ouvrir un notebook` → `GitHub` → coller l’URL du repo.

### Étape 2 – Activer le GPU (OBLIGATOIRE pour RAPIDS)
> **⚠️ Sans GPU, RAPIDS ne peut pas s’installer et le notebook bascule sur CPU (scikit‑learn).**

`Modifier` → `Paramètres du notebook` → **Accélérateur matériel** → **GPU (T4 recommandé)**

### Étape 3 – Lancer toutes les cellules
`Exécution` → `Tout exécuter` (ou `Ctrl+F9`)

> ⏱️ **Durée estimée** : ~15–25 min (dont ~10 min pour l’installation RAPIDS)

---

## 🔑 Authentification Kaggle (obligatoire pour le dataset réel)

Le notebook télécharge automatiquement le jeu de données depuis Kaggle. Pour cela, **vous devez fournir votre clé d’API Kaggle** de l’une des deux manières suivantes :

### Utiliser le fichier `access_token` (recommandé)
1. Téléchargez votre fichier `access_token` depuis votre compte Kaggle (généré automatiquement).
2. Lors de l’exécution de la **Cellule 4**, le notebook vous demandera de téléverser ce fichier.
3. Saisissez votre **nom d’utilisateur Kaggle** si nécessaire.
4. Le notebook configure automatiquement l’API Kaggle et télécharge le dataset.

> 💡 **Sécurité** : votre clé n’est jamais exposée – elle est stockée temporairement dans l’environnement Colab et supprimée à la fin de la session.

---

## 📦 Prérequis

### Environnement
- Google Colab avec GPU activé (T4 ou K80)
- Python 3.10+ (fourni par Colab)
- Connexion Internet (pour télécharger RAPIDS, le dataset et les bibliothèques)
- **Compte Kaggle** (gratuit) pour obtenir la clé d’API

### Bibliothèques installées automatiquement

| Bibliothèque | Version | Usage |
|---|---|---|
| `cudf`, `cuml` | ≥24.x | DataFrames GPU et ML (RAPIDS) |
| `xgboost` | ≥2.0 | Gradient Boosting avec accélération GPU |
| `shap` | ≥0.45 | Explicabilité SHAP |
| `matplotlib`, `seaborn` | ≥3.7, ≥0.12 | Visualisations |
| `pandas`, `numpy` | ≥2.0, ≥1.24 | Manipulation CPU et calculs |
| `scikit-learn` | ≥1.3 | Métriques, split, StandardScaler (fallback CPU) |
| `kaggle`, `kagglehub` | - | Téléchargement du dataset (installées automatiquement) |

> ℹ️ Sur Colab, RAPIDS s’installe via la méthode officielle `rapidsai/rapidsai-csp` qui gère automatiquement la version CUDA.

---

## 📊 Résultats attendus (sur le vrai dataset FAO)

### Performances typiques (test set)

| Modèle | RMSE (hg/ha) | R² |
|--------|--------------|-----|
| Random Forest (cuML) | ~15 000 | ~0.88 |
| **XGBoost (GPU)**   | **~13 500** | **~0.91** |
| Régression Linéaire (cuML) | ~28 000 | ~0.65 |

> Les valeurs exactes varient selon l’échantillon et la source du dataset. XGBoost obtient systématiquement les meilleures performances.

### Graphiques générés

| Fichier | Description |
|---------|-------------|
| `eda_fao_real.png` | Analyse exploratoire (distribution, corrélations, évolution temporelle) |
| `model_comparison.png` | Comparaison des modèles (RMSE, R², temps d’entraînement) |
| `shap_<modele>.png` | SHAP summary plots (bar + dot) pour chaque modèle réussi |
| `shap_waterfall.png` | Waterfall plot expliquant une prédiction individuelle |
| `feature_importance_builtin.png` | Importance intégrée (Random Forest / coefficients) |

---

## 🗂️ Structure du projet
agriculture-yield-gpu/
├── README.md # Ce fichier
├── requirements.txt # Dépendances (référence)
├── environment.yml # Conda environment (référence)
├── .gitignore # Fichiers à ignorer
├── agriculture_yield_pipeline.ipynb # Notebook principal (tout-en-un)
└── setup_colab.sh # Script d’installation optionnel

---

## 🔧 Fonctionnalités techniques (code robuste)

### Gestion des données
- ✅ **Chargement depuis Kaggle** : authentification via token (fichier `access_token` ou `kaggle.json`), téléchargement direct dans Colab.
- ✅ **Fallback synthétique** : si l’API Kaggle échoue (problème de token ou de réseau), un dataset synthétique réaliste est généré.
- ✅ **Prétraitement automatique** :
  - Suppression des lignes sans valeur cible
  - Imputation des NaN (médiane pour numériques, mode pour catégorielles)
  - Encodage LabelEncoder des variables catégorielles (`country`, `crop`, `region`, `season`)
  - Feature engineering : interactions `fertilizer × arable_land`, `temp × log(rainfall)`, variables cycliques sur l’année

### Modèles et GPU
- ✅ **Détection automatique du GPU** : utilisation de `cupy` pour tester la disponibilité.
- ✅ **Trois modèles** avec bascule GPU/CPU transparente :
  - Random Forest : `cuML` si GPU, sinon `scikit‑learn`
  - XGBoost : `device='cuda'` si GPU, sinon `tree_method='hist'` (CPU)
  - Régression Linéaire : `cuML` si GPU, sinon `scikit‑learn`
- ✅ **Normalisation** : `StandardScaler` appliqué uniquement pour la régression linéaire.

### SHAP et explicabilité
- ✅ **TreeExplainer** pour Random Forest et XGBoost (après conversion si nécessaire)
- ✅ **LinearExplainer** pour la régression linéaire (ou KernelExplainer en fallback)
- ✅ **Génération automatique** des summary plots (bar + dot) et waterfall pour le meilleur modèle.

### Visualisations
- ✅ Matrice de corrélation, évolution temporelle, distribution du rendement
- ✅ Comparaison des performances (RMSE, R², temps d’entraînement)
- ✅ Sauvegarde de tous les graphiques en `.png` dans le répertoire courant.

---

## ⚠️ Notes importantes

1. **Redémarrage Colab** : L’installation de RAPIDS peut nécessiter un redémarrage du kernel. Le notebook détecte cela et vous guide.
2. **Quota GPU Colab** : Le GPU gratuit est limité (~12h/jour). En cas d’expiration, relancez le notebook.
3. **Version CUDA** : Le script d’installation détecte automatiquement la version CUDA de l’instance.
4. **Dataset Kaggle** : La première exécution vous demandera de télécharger votre fichier `access_token` ou `kaggle.json`. Conservez‑le dans votre espace Colab le temps de la session.
5. **Erreur de token** : Si le téléchargement échoue, le notebook bascule sur un dataset synthétique (pour que le projet reste fonctionnel).

---

## 📚 Références

- [RAPIDS Documentation](https://docs.rapids.ai/)
- [FAO FAOSTAT – site officiel](https://www.fao.org/faostat/en/#data)
- [Kaggle Dataset utilisé](https://www.kaggle.com/datasets/vijayveersingh/faostat-crops-and-livestock-data)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [cuML User Guide](https://docs.rapids.ai/api/cuml/stable/)

---

## 📝 Licence

MIT License – voir [LICENSE](LICENSE) pour les détails.

---

*Projet réalisé dans le cadre d’une démonstration de ML agricole avec accélération GPU et explicabilité SHAP.*