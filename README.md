# 🌾 Prédiction du Rendement Agricole avec RAPIDS cuML + SHAP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/votre-username/agriculture-yield-gpu/blob/main/v3_pro_max_ultra_plus.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-cuML-green.svg)](https://rapids.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

Pipeline ML complet pour prédire le **rendement des cultures agricoles** (hg/ha) à partir des **données historiques de la FAO (FAOSTAT)**. Le notebook exploite l’accélération GPU via **RAPIDS cuML** pour entraîner trois modèles :

- 🌲 **Random Forest** (cuML / scikit‑learn fallback)
- ⚡ **XGBoost** (avec support GPU `device='cuda'`)
- 📈 **Régression Linéaire** (cuML / scikit‑learn fallback)

L’explicabilité est assurée par **SHAP** (SHapley Additive exPlanations) avec visualisations interactives (summary plots bar + dot, waterfall plot).

---

## 🎯 Objectifs

| Objectif | Détail |
|----------|--------|
| **Dataset** | FAO FAOSTAT (Kaggle : `patelris/crop-yield-prediction-dataset`) – 28 242 lignes |
| **Cible** | Rendement en hg/ha (`yield_hg_ha`) |
| **Métriques** | RMSE, R² (train & test) |
| **GPU** | NVIDIA T4 / K80 (Google Colab gratuit) |
| **Explicabilité** | SHAP summary plots (bar + dot) + waterfall |

---

## 🚀 Instructions d’exécution sur Google Colab

### Étape 1 – Ouvrir le notebook
1. Cliquer sur le badge **Open in Colab** ci-dessus, **OU**
2. Aller sur [colab.research.google.com](https://colab.research.google.com/) → `Fichier` → `Ouvrir un notebook` → `GitHub` → coller l’URL du repo.

### Étape 2 – Activer le GPU (OBLIGATOIRE pour RAPIDS)
> **⚠️ Sans GPU, RAPIDS ne peut pas s’installer et le notebook bascule sur CPU (scikit‑learn).**

`Modifier` → `Paramètres du notebook` → **Accélérateur matériel** → **GPU (T4 recommandé)**

### Étape 3 – Lancer toutes les cellules
`Exécution` → `Tout exécuter` (ou `Ctrl+F9`)

> ⏱️ **Durée estimée** : ~15–20 min (dont ~10 min pour l’installation RAPIDS)

---

## 🔑 Authentification Kaggle (obligatoire pour le dataset réel)

Le notebook télécharge automatiquement le jeu de données depuis Kaggle. Pour cela, **vous devez fournir votre clé d’API Kaggle** :

1. Rendez-vous sur [kaggle.com → Settings → API → Create New Token](https://www.kaggle.com/settings)
2. Téléchargez le fichier `kaggle.json`
3. Dans la **Cellule 4** du notebook, cliquez sur **« Choisir des fichiers »** et sélectionnez ce fichier `kaggle.json`.
4. Saisissez votre **nom d’utilisateur Kaggle** lorsqu’il vous est demandé.

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
| `cudf`, `cuml` | 26.02+ | DataFrames GPU et ML (RAPIDS) |
| `xgboost` | ≥2.0 | Gradient Boosting avec accélération GPU |
| `shap` | ≥0.45 | Explicabilité SHAP |
| `matplotlib` | ≥3.7 | Visualisations |
| `pandas`, `numpy` | ≥2.0, ≥1.24 | Manipulation CPU et calculs |
| `scikit-learn` | ≥1.3 | Métriques, split, StandardScaler (fallback CPU) |
| `kaggle` | - | Téléchargement du dataset |

> ℹ️ Sur Colab, RAPIDS s’installe via la méthode officielle `rapidsai/rapidsai-csp-utils` qui gère automatiquement la version CUDA.

---

## 📊 Résultats attendus (sur le vrai dataset FAO)

### Performances typiques (test set) – extrait du notebook

| Modèle | RMSE (hg/ha) | R² | Temps (s) |
|--------|--------------|-----|-----------|
| Random Forest (cuML) | ~10 127 | ~0.986 | 1.1 |
| **XGBoost (GPU)**   | **~11 081** | **~0.983** | 1.1 |
| Régression Linéaire (cuML) | ~82 452 | ~0.063 | 0.4 |

> Les performances peuvent varier légèrement selon l’échantillonnage et la version des librairies. Random Forest obtient ici le meilleur R², XGBoost reste très proche.

### Graphiques générés

| Fichier | Description |
|---------|-------------|
| `eda_fao.png` | Analyse exploratoire (histogrammes des variables) |
| `model_comparison.png` | Comparaison des modèles (RMSE, R², temps d’entraînement) |
| `shap_<modele>.png` | SHAP summary plots (bar + dot) pour XGBoost |
| `shap_waterfall.png` | Waterfall plot expliquant une prédiction individuelle |

---

## 🗂️ Structure du projet

agriculture-yield-gpu/
├── README.md # Ce fichier
├── v3_pro_max_ultra_plus.ipynb # Notebook principal
├── requirements.txt # Dépendances (optionnel)
├── .gitignore # Fichiers à ignorer
└── setup_colab.sh # Script d’installation (optionnel)


---

## 🔧 Fonctionnalités techniques (code robuste)

### Gestion des données
- ✅ **Chargement depuis Kaggle** : authentification via `kaggle.json` téléversé, téléchargement direct du dataset.
- ✅ **Nettoyage automatique** :
  - Renommage des colonnes (`Area` → `country`, `Item` → `crop`, `Year` → `year`, `hg/ha_yield` → `yield_hg_ha`)
  - Suppression des doublons et lignes avec valeurs manquantes sur les colonnes clés.
  - Filtrage des rendements > 0.
- ✅ **Feature engineering** :
  - Encodage `LabelEncoder` des variables catégorielles (`country`, `crop`)
  - Ajout de `year_trend` (année normalisée)
  - Interaction `crop_year = crop * year_trend`

### Modèles et GPU
- ✅ **Détection automatique du GPU** : tentative d’import de `cudf` et `cuml` ; bascule transparente vers scikit‑learn CPU si non disponibles.
- ✅ **Trois modèles** :
  - Random Forest : `cuML` si GPU, sinon `sklearn` avec `n_jobs=-1`
  - XGBoost : `device='cuda'` si GPU, sinon `tree_method='hist'` (CPU)
  - Régression Linéaire : `cuML` si GPU, sinon `sklearn`
- ✅ **Normalisation** : `StandardScaler` appliqué uniquement pour la régression linéaire.

### SHAP et explicabilité
- ✅ **TreeExplainer** pour XGBoost (compatible GPU/CPU)
- ✅ Génération des summary plots (bar + dot) et waterfall plot.
- ✅ Sauvegarde des visualisations en `png` pour les rapports.

### Visualisations
- ✅ Matrice de corrélation (EDA), histogrammes des variables.
- ✅ Comparaison des performances (RMSE, R², temps d’entraînement) avec barres groupées.
- ✅ Graphique `Réel vs Prédit` pour le meilleur modèle.

---

## ⚠️ Notes importantes

1. **Redémarrage Colab** : L’installation de RAPIDS ne nécessite pas de redémarrage ici (utilisation directe du script pip-install).
2. **Quota GPU Colab** : Le GPU gratuit est limité (~12h/jour). En cas d’expiration, relancez le notebook.
3. **Version CUDA** : Le script d’installation détecte automatiquement la version CUDA de l’instance.
4. **Dataset Kaggle** : La première exécution vous demandera de télécharger votre fichier `kaggle.json`. Conservez‑le dans votre espace Colab le temps de la session.
5. **Erreur de token** : Si le téléchargement échoue, le notebook ne bascule pas sur un dataset synthétique (contrairement à d’autres versions) ; assurez-vous que votre fichier `kaggle.json` est valide.

---

## 📚 Références

- [RAPIDS Documentation](https://docs.rapids.ai/)
- [FAO FAOSTAT – site officiel](https://www.fao.org/faostat/en/#data)
- [Kaggle Dataset utilisé](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [cuML User Guide](https://docs.rapids.ai/api/cuml/stable/)

---

## 📝 Licence

MIT License – voir [LICENSE](LICENSE) pour les détails.

---

*Projet réalisé dans le cadre d’une démonstration de ML agricole avec accélération GPU et explicabilité SHAP.*