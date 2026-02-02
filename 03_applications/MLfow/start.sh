#!/bin/bash
# =====================================================================
# Script de Démarrage - MLflow sur HuggingFace Spaces
# =====================================================================

set -e

echo "========================================="
echo "🚀 MLflow Server - HuggingFace Spaces"
echo "========================================="
echo ""

# ========================================
# VÉRIFIER VARIABLES D'ENVIRONNEMENT
# ========================================

echo "🔍 Vérification des variables d'environnement..."

if [ -z "$NEON_DATABASE_URL" ]; then
    echo "❌ ERROR: NEON_DATABASE_URL not set"
    echo "   → Configure in HF Space Settings → Repository secrets"
    exit 1
fi

if [ -z "$R2_ACCESS_KEY_ID" ]; then
    echo "❌ ERROR: R2_ACCESS_KEY_ID not set"
    exit 1
fi

if [ -z "$R2_SECRET_ACCESS_KEY" ]; then
    echo "❌ ERROR: R2_SECRET_ACCESS_KEY not set"
    exit 1
fi

if [ -z "$R2_ENDPOINT_URL" ]; then
    echo "❌ ERROR: R2_ENDPOINT_URL not set"
    exit 1
fi

if [ -z "$R2_BUCKET_NAME" ]; then
    echo "❌ ERROR: R2_BUCKET_NAME not set"
    exit 1
fi

echo "✅ Toutes les variables d'environnement sont définies"
echo ""

# ========================================
# CONFIGURATION MLFLOW
# ========================================

# # Backend Store URI (Neon PostgreSQL)
export BACKEND_STORE_URI="${NEON_DATABASE_URL}"

# Artifact Root (Cloudflare R2)
export ARTIFACT_ROOT="s3://${R2_BUCKET_NAME}/mlflow/artifacts"

# AWS/R2 Configuration
export AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}"
export AWS_ENDPOINT_URL="${R2_ENDPOINT_URL}"
export AWS_DEFAULT_REGION="auto"

# HuggingFace Spaces utilise le port 7860
export MLFLOW_PORT="${PORT:-7860}"
export MLFLOW_HOST="0.0.0.0"

# ========================================
# AFFICHER CONFIGURATION
# ========================================

echo "📊 Configuration MLflow:"
echo "  Backend Store: MotherDuck (DuckDB cloud)"
echo "  Artifact Store: Cloudflare R2"
echo "  Host: ${MLFLOW_HOST}"
echo "  Port: ${MLFLOW_PORT}"
echo ""

# ========================================
# DÉMARRER MLFLOW SERVER
# ========================================

echo "🚀 Démarrage du serveur MLflow..."
echo ""

exec mlflow server \
    --backend-store-uri "${BACKEND_STORE_URI}" \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --host "${MLFLOW_HOST}" \
    --port "${MLFLOW_PORT}" \
    --gunicorn-opts "--timeout=300 --workers=2"
