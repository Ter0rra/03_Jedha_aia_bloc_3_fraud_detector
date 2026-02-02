"""
🚨 Fraud Detection API - Level UP Edition
=========================================

API FastAPI pour la détection de fraude en temps réel
avec preprocessing et feature engineering

Fonctionnalités:
- Download automatique du model + preprocessor depuis HuggingFace
- 3 endpoints: /predict, /preprocess, /feat_eng
- Feature engineering complet (distance GPS, features temporelles, âge)
- Documentation interactive sur /docs

Author: Terorra
Date: January 2026
Version: 2.0.0
"""

# =====================================================================
# IMPORTS
# =====================================================================

# FastAPI et types
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# HuggingFace pour télécharger les modèles
from huggingface_hub import hf_hub_download

# ML et data
import joblib
import pandas as pd
import numpy as np

# Utilitaires
import os
from datetime import datetime
import time

# Notre module de feature engineering
from feature_engineering import (
    engineer_features,
    prepare_for_model,
    get_model_features,
    haversine_distance,
    extract_time_features
)


# =====================================================================
# CONFIGURATION GLOBALE
# =====================================================================

# Repository HuggingFace où sont stockés les modèles
REPO_ID = "Terorra/fd_model_jedha"

# Noms des fichiers sur HuggingFace
MODEL_FILENAME = "fraud_model.pkl"           # Le modèle RandomForest
PREPROCESSOR_FILENAME = "preprocessor.pkl"   # Le preprocessor (ColumnTransformer)

# Version du modèle (None = latest, ou "v1", "v2", etc.)
MODEL_VERSION = None


# =====================================================================
# VARIABLES GLOBALES (modèles chargés en mémoire)
# =====================================================================

# Ces variables seront remplies au démarrage de l'API
model = None         # Le modèle ML (RandomForestClassifier)
preprocessor = None  # Le preprocessor (StandardScaler + OneHotEncoder)


# =====================================================================
# CRÉATION DE L'APPLICATION FASTAPI
# =====================================================================

app = FastAPI(
    # Titre qui apparaît dans la doc
    title="🚨 Fraud Detection API - Level UP",
    
    # Description complète (supporte Markdown)
    description="""
    # API de Détection de Fraude en Temps Réel
    
    Cette API utilise le Machine Learning pour détecter les transactions frauduleuses
    sur les cartes de crédit.
    
    ## 🚀 Fonctionnalités
    
    ### Endpoints Principaux
    
    1. **`/predict`** - Prédiction complète
       - Prend les données brutes
       - Applique le feature engineering
       - Applique le preprocessing
       - Retourne la prédiction de fraude
    
    2. **`/feat_eng`** - Feature Engineering seulement
       - Calcule la distance GPS client-marchand
       - Extrait les features temporelles (heure, jour, weekend, etc.)
       - Calcule l'âge du porteur
       - Retourne les features transformées
    
    3. **`/preprocess`** - Preprocessing seulement
       - Prend les features (déjà engineered)
       - Applique StandardScaler (normalisation)
       - Applique OneHotEncoder (encoding catégories)
       - Retourne les features preprocessed (prêtes pour le modèle)
    
    ### Endpoints Utilitaires
    
    - **`/health`** - Vérifier que l'API fonctionne
    - **`/model/info`** - Informations sur le modèle ML
    - **`/features`** - Liste des features nécessaires
    
    ## 📊 Workflow Complet
    
    ```
    Données Brutes
        ↓
    /feat_eng → Feature Engineering
        ↓
    /preprocess → Preprocessing (scaling + encoding)
        ↓
    /predict → Prédiction ML
        ↓
    Résultat: Fraude ou Non
    ```
    
    ## 🎯 Modèle ML
    
    - **Algorithme**: RandomForestClassifier
    - **Recall**: > 90% (optimisé pour détecter les fraudes)
    - **Features**: 21 features (17 numériques + 4 catégorielles)
    - **Preprocessing**: StandardScaler + OneHotEncoder
    - **Hébergement**: HuggingFace Hub
    
    ## 💡 Cas d'Usage
    
    1. **Validation en temps réel**: Valider une transaction au moment du paiement
    2. **Analyse batch**: Analyser des milliers de transactions historiques
    3. **Monitoring**: Surveiller les patterns de fraude
    4. **Reporting**: Générer des rapports de fraude
    
    ## 🔧 Feature Engineering
    
    L'API calcule automatiquement:
    - **distance_km**: Distance GPS entre client et marchand (formule Haversine)
    - **hour**: Heure de la transaction (0-23)
    - **is_night, is_morning, is_afternoon, is_evening**: Période de la journée
    - **is_business_hour**: Transaction pendant heures de bureau (8h-17h)
    - **is_weekend**: Transaction le weekend
    - **age**: Âge du porteur de carte
    - **year, month, day, dayofweek**: Composantes de la date
    
    ## 📚 Documentation
    
    - Cette page: Documentation interactive avec exemples
    - Essayez les endpoints directement depuis cette page!
    - Chaque endpoint a des exemples pré-remplis
    
    ## 🎓 Pour Commencer
    
    1. Testez `/health` pour vérifier que l'API fonctionne
    2. Regardez `/features` pour voir les features nécessaires
    3. Essayez `/feat_eng` avec des données de test
    4. Utilisez `/predict` pour une prédiction complète
    """,
    
    version="2.0.0",
    
    contact={
        "name": "Terorra",
        "email": "your.email@example.com",
    },
    
    license_info={
        "name": "MIT",
    },
    
    # Tags pour organiser les endpoints dans la doc
    openapi_tags=[
        {
            "name": "🎯 Prediction",
            "description": "Endpoints de prédiction de fraude"
        },
        {
            "name": "🔧 Feature Engineering",
            "description": "Transformation des features"
        },
        {
            "name": "⚙️ Preprocessing",
            "description": "Preprocessing des données"
        },
        {
            "name": "📊 Information",
            "description": "Informations sur l'API et le modèle"
        },
    ]
)


# =====================================================================
# SCHEMAS PYDANTIC (Définition des types de données)
# =====================================================================

class TransactionRawInput(BaseModel):
    """
    Données BRUTES d'une transaction (avant feature engineering)
    
    Ce sont les données telles qu'elles arrivent de la base de données
    ou du système de paiement, SANS transformation.
    """
    # Informations carte
    cc_num: int = Field(
        ...,
        description="Numéro de carte de crédit (hashé)",
        example=374125201044065
    )
    
    # Montant
    amt: float = Field(
        ...,
        description="Montant de la transaction en dollars",
        example=150.75,
        gt=0
    )
    
    # Localisation client
    lat: float = Field(
        ...,
        description="Latitude du client (coordonnées GPS)",
        example=40.7128,
        ge=-90,
        le=90
    )
    long: float = Field(
        ...,
        description="Longitude du client (coordonnées GPS)",
        example=-74.0060,
        ge=-180,
        le=180
    )
    
    # Ville
    city_pop: int = Field(
        ...,
        description="Population de la ville du client",
        example=8000000,
        gt=0
    )
    zip: int = Field(
        ...,
        description="Code postal",
        example=10001
    )
    
    # Localisation marchand
    merch_lat: float = Field(
        ...,
        description="Latitude du marchand (coordonnées GPS)",
        example=40.7589,
        ge=-90,
        le=90
    )
    merch_long: float = Field(
        ...,
        description="Longitude du marchand (coordonnées GPS)",
        example=-73.9851,
        ge=-180,
        le=180
    )
    
    # Marchand
    merchant: str = Field(
        ...,
        description="Nom du marchand",
        example="Amazon"
    )
    category: str = Field(
        ...,
        description="Catégorie de transaction",
        example="shopping_net"
    )
    
    # Client
    gender: str = Field(
        ...,
        description="Genre du client (M/F)",
        example="M"
    )
    state: str = Field(
        ...,
        description="État (US)",
        example="NY"
    )
    dob: str = Field(
        ...,
        description="Date de naissance (YYYY-MM-DD)",
        example="1990-01-15"
    )
    
    # Transaction
    transaction_time: str = Field(
        ...,
        description="Heure de la transaction (YYYY-MM-DD HH:MM:SS)",
        example="2026-01-29 14:30:00"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "cc_num": 374125201044065,
                "amt": 150.75,
                "lat": 40.7128,
                "long": -74.0060,
                "city_pop": 8000000,
                "zip": 10001,
                "merch_lat": 40.7589,
                "merch_long": -73.9851,
                "merchant": "Amazon",
                "category": "shopping_net",
                "gender": "M",
                "state": "NY",
                "dob": "1990-01-15",
                "transaction_time": "2026-01-29 14:30:00"
            }
        }


class FeaturesEngineeredOutput(BaseModel):
    """
    Résultat du Feature Engineering
    
    Contient les données originales + les features calculées
    """
    # Données originales
    original_data: Dict[str, Any] = Field(
        ...,
        description="Données brutes d'entrée"
    )
    
    # Features engineered
    engineered_features: Dict[str, Any] = Field(
        ...,
        description="Nouvelles features calculées"
    )
    
    # Toutes les features combinées
    all_features: Dict[str, Any] = Field(
        ...,
        description="Données originales + features engineered"
    )


class PreprocessedOutput(BaseModel):
    """
    Résultat du Preprocessing
    
    Features transformées (scaled + encoded) prêtes pour le modèle
    """
    preprocessed_shape: tuple = Field(
        ...,
        description="Dimensions des données preprocessed (lignes, colonnes)"
    )
    
    sample_values: List[float] = Field(
        ...,
        description="Premières valeurs (pour debug)"
    )
    
    message: str = Field(
        ...,
        description="Message de confirmation"
    )


class PredictionOutput(BaseModel):
    """
    Résultat de la Prédiction de Fraude
    """
    # Prédiction
    is_fraud: bool = Field(
        ...,
        description="True si la transaction est frauduleuse"
    )
    
    fraud_probability: float = Field(
        ...,
        description="Probabilité de fraude (0.0 à 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Classification du risque
    risk_level: str = Field(
        ...,
        description="Niveau de risque: LOW, MEDIUM, HIGH, CRITICAL"
    )
    
    # Confiance du modèle
    confidence: float = Field(
        ...,
        description="Confiance du modèle (0.0 à 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Métadonnées
    timestamp: str = Field(
        ...,
        description="Heure de la prédiction (ISO format)"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Temps de traitement en millisecondes"
    )


# =====================================================================
# FONCTIONS HELPER
# =====================================================================

def load_models_from_hf():
    """
    Télécharge et charge les modèles depuis HuggingFace Hub
    
    Cette fonction:
    1. Télécharge fraud_model.pkl (le modèle ML)
    2. Télécharge preprocessor.pkl (le preprocessor)
    3. Charge les 2 fichiers en mémoire
    4. Met à jour les variables globales model et preprocessor
    
    Returns:
        tuple: (success: bool, message: str)
            success = True si tout s'est bien passé
            message = Message d'information ou d'erreur
    """
    global model, preprocessor
    
    try:
        print("=" * 70)
        print("📥 Téléchargement des modèles depuis HuggingFace...")
        print(f"   Repository: {REPO_ID}")
        print("=" * 70)
        
        # ========================================
        # 1. TÉLÉCHARGER LE MODÈLE ML
        # ========================================
        
        print(f"\n⬇️ Download: {MODEL_FILENAME}...")
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME, # None = latest
            cache_dir="/tmp"         # Dossier de cache
        )
        print(f"✅ Téléchargé: {model_path}")
        
        # Charger le modèle
        model = joblib.load(model_path)
        print(f"✅ Modèle chargé: {type(model).__name__}")
        
        # ========================================
        # 2. TÉLÉCHARGER LE PREPROCESSOR
        # ========================================
        
        print(f"\n⬇️ Download: {PREPROCESSOR_FILENAME}...")
        preprocessor_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=PREPROCESSOR_FILENAME,
            cache_dir="/tmp"
        )
        print(f"✅ Téléchargé: {preprocessor_path}")
        
        # Charger le preprocessor
        preprocessor = joblib.load(preprocessor_path)
        print(f"✅ Preprocessor chargé: {type(preprocessor).__name__}")
        
        print("\n" + "=" * 70)
        print("✅ TOUS LES MODÈLES SONT CHARGÉS ET PRÊTS")
        print("=" * 70)
        
        return True, "Models loaded successfully"
        
    except Exception as e:
        error_msg = f"Erreur lors du chargement des modèles: {str(e)}"
        print(f"\n❌ {error_msg}")
        return False, error_msg


def calculate_risk_level(probability: float) -> str:
    """
    Calcule le niveau de risque basé sur la probabilité de fraude
    
    Args:
        probability (float): Probabilité de fraude (0.0 à 1.0)
    
    Returns:
        str: Niveau de risque (LOW, MEDIUM, HIGH, CRITICAL)
    
    Seuils:
        < 0.3  : LOW       (Risque faible)
        < 0.6  : MEDIUM    (Risque moyen)
        < 0.8  : HIGH      (Risque élevé)
        >= 0.8 : CRITICAL  (Risque critique)
    """
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    elif probability < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"


# =====================================================================
# ÉVÉNEMENT DE DÉMARRAGE
# =====================================================================

@app.on_event("startup")
async def startup_event():
    """
    Fonction appelée AU DÉMARRAGE de l'API
    
    Cette fonction:
    - Est exécutée UNE SEULE FOIS quand l'API démarre
    - Télécharge et charge les modèles en mémoire
    - Les modèles restent en mémoire pour toutes les requêtes
    
    Si les modèles ne se chargent pas, l'API démarre quand même
    mais les endpoints de prédiction renverront une erreur 503.
    """
    print("\n" + "🚀" * 35)
    print("🚀 DÉMARRAGE DE L'API FRAUD DETECTION")
    print("🚀" * 35)
    
    # Charger les modèles
    success, message = load_models_from_hf()
    
    if success:
        print("\n✅ API prête à recevoir des requêtes!\n")
    else:
        print(f"\n⚠️ API démarrée mais modèles non chargés: {message}")
        print("⚠️ Les endpoints de prédiction ne fonctionneront pas.\n")


# =====================================================================
# ENDPOINTS - INFORMATION
# =====================================================================

@app.get(
    "/",
    tags=["📊 Information"],
    summary="Page d'accueil",
    description="Informations générales sur l'API"
)
async def root():
    """
    Endpoint racine - Informations sur l'API
    
    Retourne:
    - Nom de l'API
    - Version
    - Liens vers la documentation
    - Liste des endpoints disponibles
    """
    return {
        "message": "🚨 Fraud Detection API - Level UP",
        "version": "2.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "prediction": {
                "predict": "/predict - Prédiction complète (feat_eng + preprocess + predict)",
            },
            "feature_engineering": {
                "feat_eng": "/feat_eng - Feature engineering seulement",
            },
            "preprocessing": {
                "preprocess": "/preprocess - Preprocessing seulement",
            },
            "information": {
                "model_info": "/model/info - Informations sur le modèle",
                "features": "/features - Liste des features nécessaires",
            }
        },
        "example_workflow": {
            "1": "Données brutes → /feat_eng → Features engineered",
            "2": "Features engineered → /preprocess → Features preprocessed",
            "3": "Features preprocessed → /predict → Prédiction",
            "shortcut": "Données brutes → /predict → Prédiction directe (recommandé)"
        }
    }


@app.get(
    "/health",
    tags=["📊 Information"],
    summary="Health check",
    description="Vérifier que l'API et les modèles sont opérationnels"
)
async def health_check():
    """
    Vérifie l'état de santé de l'API
    
    Retourne:
    - Status de l'API (healthy/unhealthy)
    - État du modèle ML (loaded/not loaded)
    - État du preprocessor (loaded/not loaded)
    - Timestamp
    """
    # Vérifier si les modèles sont chargés
    models_loaded = (model is not None) and (preprocessor is not None)
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_repo": REPO_ID,
        "model_type": type(model).__name__ if model else None,
        "preprocessor_type": type(preprocessor).__name__ if preprocessor else None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get(
    "/model/info",
    tags=["📊 Information"],
    summary="Informations sur le modèle",
    description="Détails techniques sur le modèle ML et le preprocessor"
)
async def model_info():
    """
    Informations détaillées sur le modèle ML
    
    Retourne:
    - Type de modèle
    - Repository HuggingFace
    - Nombre de features
    - Liste des features
    """
    # Vérifier que les modèles sont chargés
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please check /health endpoint."
        )
    
    # Récupérer la liste des features
    features = get_model_features()
    
    return {
        "model": {
            "type": type(model).__name__,
            "repo_id": REPO_ID,
            "filename": MODEL_FILENAME,
            "version": MODEL_VERSION or "latest"
        },
        "preprocessor": {
            "type": type(preprocessor).__name__,
            "filename": PREPROCESSOR_FILENAME
        },
        "features": {
            "total": len(features),
            "numerical": 17,
            "categorical": 4,
            "list": features
        }
    }


@app.get(
    "/features",
    tags=["📊 Information"],
    summary="Liste des features",
    description="Liste complète des features nécessaires pour une prédiction"
)
async def list_features():
    """
    Liste toutes les features attendues par le modèle
    
    Retourne:
    - Features numériques (17)
    - Features catégorielles (4)
    - Total (21)
    """
    features = get_model_features()
    
    numerical = features[:17]  # Premières 17 = numériques
    categorical = features[17:]  # Dernières 4 = catégorielles
    
    return {
        "total_features": len(features),
        "numerical_features": {
            "count": len(numerical),
            "list": numerical
        },
        "categorical_features": {
            "count": len(categorical),
            "list": categorical
        },
        "all_features_in_order": features
    }


# =====================================================================
# ENDPOINTS - FEATURE ENGINEERING
# =====================================================================

@app.post(
    "/feat_eng",
    response_model=FeaturesEngineeredOutput,
    tags=["🔧 Feature Engineering"],
    summary="Feature Engineering",
    description="Transforme les données brutes en features pour le modèle ML"
)
async def feature_engineering_endpoint(transaction: TransactionRawInput):
    """
    Applique le FEATURE ENGINEERING sur une transaction
    
    ## Ce que fait cet endpoint:
    
    1. **Calcul de distance GPS**
       - Calcule la distance entre le client et le marchand
       - Utilise la formule Haversine (précision: ±1%)
       - Feature créée: `distance_km`
    
    2. **Extraction des features temporelles**
       - Heure de la journée (0-23)
       - Jour de la semaine (0-6)
       - Période (nuit, matin, après-midi, soir)
       - Weekend ou non
       - Heures de bureau ou non
       - Features créées: `hour`, `dayofweek`, `is_night`, `is_morning`, 
         `is_afternoon`, `is_evening`, `is_business_hour`, `is_we`, 
         `year`, `month`, `day`
    
    3. **Calcul de l'âge**
       - À partir de la date de naissance
       - Feature créée: `age`
    
    ## Input:
    Données brutes de la transaction (voir schema TransactionRawInput)
    
    ## Output:
    - `original_data`: Données brutes d'entrée
    - `engineered_features`: Nouvelles features calculées
    - `all_features`: Toutes les features (original + engineered)
    
    ## Exemple d'utilisation:
    ```python
    import requests
    
    data = {
        "cc_num": 374125201044065,
        "amt": 150.75,
        "lat": 40.7128,
        "long": -74.0060,
        # ... autres champs
    }
    
    response = requests.post("http://localhost:8000/feat_eng", json=data)
    features = response.json()["all_features"]
    ```
    """
    try:
        # Convertir en dictionnaire
        transaction_dict = transaction.dict()
        
        print("\n" + "=" * 70)
        print("🔧 FEATURE ENGINEERING")
        print("=" * 70)
        
        # Appliquer le feature engineering
        # (voir feature_engineering.py pour les détails)
        engineered = engineer_features(transaction_dict)
        
        # Identifier les features qui ont été ajoutées
        original_keys = set(transaction_dict.keys())
        all_keys = set(engineered.keys())
        new_features = all_keys - original_keys
        
        print(f"\n✅ Feature engineering terminé")
        print(f"   Features ajoutées: {len(new_features)}")
        print(f"   Total features: {len(engineered)}")
        
        # Préparer la réponse
        return {
            "original_data": transaction_dict,
            "engineered_features": {k: engineered[k] for k in new_features},
            "all_features": engineered
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature engineering failed: {str(e)}"
        )


# =====================================================================
# ENDPOINTS - PREPROCESSING
# =====================================================================

@app.post(
    "/preprocess",
    response_model=PreprocessedOutput,
    tags=["⚙️ Preprocessing"],
    summary="Preprocessing",
    description="Applique le preprocessing (scaling + encoding) sur les features"
)
async def preprocessing_endpoint(features: Dict[str, Any]):
    """
    Applique le PREPROCESSING sur les features
    
    ## Ce que fait cet endpoint:
    
    1. **StandardScaler** (normalisation)
       - Met les features numériques à l'échelle
       - Moyenne = 0, Écart-type = 1
       - Exemple: 100$ → 0.52, 5000$ → 2.31
    
    2. **OneHotEncoder** (encoding catégoriel)
       - Convertit les catégories en colonnes binaires
       - Exemple: 'NY' → [0, 0, 1, 0, ...] (vecteur de 50 dimensions)
       - Exemple: 'shopping_net' → [0, 1, 0, ...] (vecteur de 14 dimensions)
    
    ## Input:
    Dictionnaire avec toutes les features (déjà engineered)
    
    Les 21 features attendues:
    - **Numériques** (17): cc_num, amt, zip, city_pop, distance_km, age, 
      hour, is_night, is_morning, is_afternoon, is_evening, is_business_hour, 
      year, month, day, dayofweek, is_we
    - **Catégorielles** (4): merchant, category, gender, state
    
    ## Output:
    - `preprocessed_shape`: Dimensions des données transformées
    - `sample_values`: Premières valeurs (pour vérification)
    - `message`: Message de confirmation
    
    ## Note:
    Les données preprocessed ne sont PAS retournées en entier 
    (trop volumineuses), seulement leur shape et un échantillon.
    
    Pour obtenir une prédiction, utilisez directement `/predict`
    qui fait feat_eng + preprocess + predict.
    """
    # Vérifier que le preprocessor est chargé
    if preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Preprocessor not loaded"
        )
    
    try:
        print("\n" + "=" * 70)
        print("⚙️ PREPROCESSING")
        print("=" * 70)
        
        # Préparer les features pour le modèle
        # (sélectionne les bonnes colonnes dans le bon ordre)
        df = prepare_for_model(features)
        
        if df is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required features. Use /features to see the full list."
            )
        
        print(f"\n📊 Features préparées: {df.shape}")
        
        # Appliquer le preprocessing
        # Le preprocessor fait:
        # 1. StandardScaler sur les numériques
        # 2. OneHotEncoder sur les catégorielles
        X_preprocessed = preprocessor.transform(df)
        
        print(f"✅ Preprocessing terminé: {X_preprocessed.shape}")
        print(f"   Input: {df.shape[1]} features")
        print(f"   Output: {X_preprocessed.shape[1]} features (après encoding)")
        
        # Retourner les informations (pas les données complètes, trop volumineux)
        return {
            "preprocessed_shape": X_preprocessed.shape,
            "sample_values": X_preprocessed[0, :10].tolist(),  # 10 premières valeurs
            "message": f"Preprocessing successful. Shape: {X_preprocessed.shape}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {str(e)}"
        )


# =====================================================================
# ENDPOINTS - PREDICTION
# =====================================================================

@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["🎯 Prediction"],
    summary="Prédiction complète",
    description="Prédiction de fraude complète (feature engineering + preprocessing + ML)"
)
async def predict_fraud(transaction: TransactionRawInput):
    """
    Prédiction COMPLÈTE de fraude
    
    ## Workflow:
    
    ```
    Données Brutes (TransactionRawInput)
        ↓
    1. Feature Engineering
        - Calcul distance GPS
        - Extraction features temporelles
        - Calcul âge
        ↓
    2. Preprocessing
        - StandardScaler (normalisation)
        - OneHotEncoder (encoding)
        ↓
    3. Prédiction ML
        - RandomForestClassifier
        - Probabilité de fraude
        ↓
    Résultat (PredictionOutput)
    ```
    
    ## Input:
    Données brutes de la transaction (voir TransactionRawInput schema)
    
    ## Output:
    - **is_fraud**: True/False - Transaction frauduleuse ou non
    - **fraud_probability**: 0.0 à 1.0 - Probabilité de fraude
    - **risk_level**: LOW/MEDIUM/HIGH/CRITICAL - Niveau de risque
    - **confidence**: 0.0 à 1.0 - Confiance du modèle
    - **timestamp**: Heure de la prédiction
    - **processing_time_ms**: Temps de traitement en millisecondes
    
    ## Niveaux de Risque:
    - **LOW**: fraud_probability < 0.3 → Transaction probablement légitime
    - **MEDIUM**: 0.3 ≤ fraud_probability < 0.6 → Vérification recommandée
    - **HIGH**: 0.6 ≤ fraud_probability < 0.8 → Transaction suspecte
    - **CRITICAL**: fraud_probability ≥ 0.8 → Bloquer la transaction
    
    ## Exemple de Code:
    
    ```python
    import requests
    
    # Données de transaction
    transaction = {
        "cc_num": 374125201044065,
        "amt": 150.75,
        "lat": 40.7128,
        "long": -74.0060,
        "city_pop": 8000000,
        "zip": 10001,
        "merch_lat": 40.7589,
        "merch_long": -73.9851,
        "merchant": "Amazon",
        "category": "shopping_net",
        "gender": "M",
        "state": "NY",
        "dob": "1990-01-15",
        "transaction_time": "2026-01-29 14:30:00"
    }
    
    # Faire la prédiction
    response = requests.post(
        "http://localhost:8000/predict",
        json=transaction
    )
    
    result = response.json()
    
    if result["is_fraud"]:
        print(f"⚠️ FRAUDE détectée! Probabilité: {result['fraud_probability']:.1%}")
        print(f"   Niveau de risque: {result['risk_level']}")
    else:
        print(f"✅ Transaction légitime. Probabilité de fraude: {result['fraud_probability']:.1%}")
    ```
    
    ## Performance:
    - Temps de traitement moyen: 10-50ms
    - Throughput: ~100-500 requêtes/seconde (selon hardware)
    
    ## Use Cases:
    1. **Validation temps réel**: Au moment du paiement
    2. **Post-transaction**: Vérification après coup
    3. **Batch processing**: Analyse de milliers de transactions
    4. **Monitoring**: Détection de patterns de fraude
    """
    # Vérifier que les modèles sont chargés
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please check /health endpoint."
        )
    
    try:
        # Timer pour mesurer le temps de traitement
        start_time = time.time()
        
        print("\n" + "🎯" * 35)
        print("🎯 PRÉDICTION DE FRAUDE - WORKFLOW COMPLET")
        print("🎯" * 35)
        
        # ========================================
        # ÉTAPE 1: FEATURE ENGINEERING
        # ========================================
        
        print("\n[1/3] 🔧 Feature Engineering...")
        transaction_dict = transaction.dict()
        engineered = engineer_features(transaction_dict)
        
        # ========================================
        # ÉTAPE 2: PREPROCESSING
        # ========================================
        
        print("\n[2/3] ⚙️ Preprocessing...")
        df = prepare_for_model(engineered)
        
        if df is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to prepare features for model"
            )
        
        # Appliquer le preprocessing
        X_preprocessed = preprocessor.transform(df)
        print(f"      Shape après preprocessing: {X_preprocessed.shape}")
        
        # ========================================
        # ÉTAPE 3: PRÉDICTION ML
        # ========================================
        
        print("\n[3/3] 🤖 Prédiction ML...")
        
        # Faire la prédiction
        prediction = model.predict(X_preprocessed)[0]  # 0 ou 1
        proba = model.predict_proba(X_preprocessed)[0]  # [proba_class_0, proba_class_1]
        
        # Extraire la probabilité de fraude (classe 1)
        fraud_prob = float(proba[1])
        
        # Calculer la confiance
        # Confiance = distance par rapport à 0.5 (seuil de décision)
        # Plus on est loin de 0.5, plus on est confiant
        confidence = abs(fraud_prob - 0.5) * 2
        
        # Calculer le niveau de risque
        risk = calculate_risk_level(fraud_prob)
        
        # Temps de traitement
        processing_time = (time.time() - start_time) * 1000  # En millisecondes
        
        # Résultat
        result = {
            "is_fraud": bool(prediction),
            "fraud_probability": round(fraud_prob, 4),
            "risk_level": risk,
            "confidence": round(confidence, 4),
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": round(processing_time, 2)
        }
        
        print("\n" + "=" * 70)
        print(f"✅ RÉSULTAT:")
        print(f"   Fraude: {result['is_fraud']}")
        print(f"   Probabilité: {result['fraud_probability']:.1%}")
        print(f"   Risque: {result['risk_level']}")
        print(f"   Temps: {result['processing_time_ms']:.2f}ms")
        print("=" * 70)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# =====================================================================
# ERROR HANDLERS (Gestion des erreurs)
# =====================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Gère les erreurs de validation de données"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid input",
            "detail": str(exc),
            "type": "ValueError"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Gère toutes les autres erreurs inattendues"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":
    """
    Lancer l'API en mode développement
    
    Commande:
        python app.py
    
    Ou avec uvicorn:
        uvicorn app:app --reload --host 0.0.0.0 --port 8000
    
    Documentation:
        http://localhost:8000/docs
    """
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en mode dev
        log_level="info"
    )
