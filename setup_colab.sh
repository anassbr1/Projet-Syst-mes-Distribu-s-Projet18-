#!/bin/bash
# ============================================================
# setup_colab.sh
# Script d'installation RAPIDS pour Google Colab
# Usage dans un notebook : !bash setup_colab.sh
# ============================================================

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║   Installation RAPIDS cuML sur Google Colab          ║"
echo "╚══════════════════════════════════════════════════════╝"

# ── 1. Vérification GPU ────────────────────────────────────
echo ""
echo "► [1/5] Vérification du GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ ERREUR : Aucun GPU détecté !"
    echo "  → Allez dans : Modifier > Paramètres du notebook > GPU"
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total \
           --format=csv,noheader,nounits
echo "✓ GPU détecté"

# ── 2. Détection version CUDA ──────────────────────────────
echo ""
echo "► [2/5] Détection de la version CUDA..."
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
echo "  Version CUDA : ${CUDA_VER:-inconnue}"

# ── 3. Installation rapidsai-csp-utils ────────────────────
echo ""
echo "► [3/5] Clonage de rapidsai-csp-utils..."
if [ -d "rapidsai-csp-utils" ]; then
    echo "  (déjà cloné, on saute)"
else
    git clone https://github.com/rapidsai/rapidsai-csp-utils.git
fi

# ── 4. Installation RAPIDS ────────────────────────────────
echo ""
echo "► [4/5] Installation de RAPIDS (peut prendre 5-10 min)..."
python rapidsai-csp-utils/colab/pip-install.py

# ── 5. Installation des dépendances supplémentaires ───────
echo ""
echo "► [5/5] Installation des dépendances Python..."
pip install -q \
    xgboost>=2.0.0 \
    shap>=0.45.0 \
    seaborn>=0.12.0 \
    tqdm>=4.65.0

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ✓ Installation terminée avec succès !              ║"
echo "║   → Redémarrez le kernel si demandé                  ║"
echo "╚══════════════════════════════════════════════════════╝"
