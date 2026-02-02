# =====================================================================
# Dockerfile - MLflow pour HuggingFace Spaces
# =====================================================================
# 
# MLflow Server avec:
# - Backend Store: MotherDuck (cloud)
# - Artifact Store: Cloudflare R2 (cloud)
# - UI accessible sur HF Spaces
# 
# Author: Terorra
# Date: January 2026
# =====================================================================

FROM python:3.12-slim

# =====================================================================
# VARIABLES D'ENVIRONNEMENT
# =====================================================================

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# =====================================================================
# INSTALLATION DÉPENDANCES SYSTÈME
# =====================================================================

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# =====================================================================
# CRÉER USER (HF Spaces utilise 'user')
# =====================================================================

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# =====================================================================
# COPIER ET INSTALLER REQUIREMENTS
# =====================================================================

# Copier requirements
COPY --chown=user:user requirements.txt .

# Installer packages Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =====================================================================
# COPIER L'APPLICATION
# =====================================================================

COPY --chown=user:user . .

# =====================================================================
# SCRIPT DE DÉMARRAGE
# =====================================================================

# Le script start.sh sera créé par nous
RUN chmod +x start.sh

# =====================================================================
# SWITCH TO USER
# =====================================================================

USER user

# =====================================================================
# EXPOSITION PORT (HF Spaces utilise $PORT ou 7860)
# =====================================================================

# HF Spaces expose automatiquement le port 7860
EXPOSE 7860

# =====================================================================
# HEALTHCHECK
# =====================================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# =====================================================================
# COMMANDE DE DÉMARRAGE
# =====================================================================

CMD ["./start.sh"]
