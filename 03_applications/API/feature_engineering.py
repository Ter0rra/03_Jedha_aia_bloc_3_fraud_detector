"""
Feature Engineering Module
--------------------------
Toutes les transformations de features pour la détection de fraude

Ce module contient les fonctions pour :
1. Calculer la distance GPS entre client et marchand
2. Extraire les features temporelles (heure, jour, weekend, etc.)
3. Calculer l'âge du porteur de carte
4. Créer toutes les features nécessaires pour le modèle ML

Author: Terorra
Date: January 2026
"""

from datetime import datetime, date
from math import radians, sin, cos, sqrt, atan2
import pandas as pd


# =====================================================================
# FONCTION 1 : CALCUL DE DISTANCE GPS
# =====================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcule la distance entre 2 points GPS avec la formule Haversine
    
    La formule Haversine permet de calculer la distance entre deux points
    sur une sphère (la Terre) à partir de leurs coordonnées GPS.
    
    Args:
        lat1 (float): Latitude du point 1 (client)
        lon1 (float): Longitude du point 1 (client)
        lat2 (float): Latitude du point 2 (marchand)
        lon2 (float): Longitude du point 2 (marchand)
    
    Returns:
        float: Distance en kilomètres (arrondie à 2 décimales)
        None: Si une coordonnée est manquante
    
    Example:
        >>> haversine_distance(48.8566, 2.3522, 51.5074, -0.1278)
        344.45  # Distance Paris-Londres en km
    """
    # Vérifier si des valeurs sont manquantes
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return None
    
    try:
        # Rayon de la Terre en kilomètres
        R = 6371
        
        # Convertir les degrés en radians (nécessaire pour les calculs trigonométriques)
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Différences de latitude et longitude
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Formule Haversine
        # a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        
        # c = 2 * atan2(√a, √(1-a))
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        # Distance = R * c
        distance = R * c
        
        # Arrondir à 2 décimales
        return round(distance, 2)
        
    except Exception as e:
        print(f"⚠️ Erreur calcul distance: {e}")
        return None


# =====================================================================
# FONCTION 2 : CALCUL DE L'ÂGE
# =====================================================================

def calculate_age(born):
    """
    Calcule l'âge à partir de la date de naissance
    
    Cette fonction calcule l'âge en années complètes, en tenant compte
    du fait que l'anniversaire peut ne pas encore être passé cette année.
    
    Args:
        born (str): Date de naissance au format 'YYYY-MM-DD'
    
    Returns:
        int: Âge en années
        None: Si la date est invalide ou manquante
    
    Example:
        >>> calculate_age('1990-01-15')
        36  # Si on est en 2026
    """
    # Vérifier si la valeur est manquante
    if pd.isna(born) or born is None or born == '':
        return None
    
    try:
        # Convertir la string en date
        born_date = datetime.strptime(str(born), '%Y-%m-%d').date()
        
        # Date du jour
        today = date.today()
        
        # Calcul de l'âge
        # On soustrait 1 si l'anniversaire n'est pas encore passé cette année
        age = today.year - born_date.year - (
            (today.month, today.day) < (born_date.month, born_date.day)
        )
        
        return age
        
    except Exception as e:
        print(f"⚠️ Erreur calcul âge pour {born}: {e}")
        return None


# =====================================================================
# FONCTION 3 : FEATURES TEMPORELLES
# =====================================================================

def extract_time_features(transaction_time):
    """
    Extrait toutes les features temporelles d'une transaction
    
    À partir de l'heure de transaction, cette fonction crée :
    - L'heure (0-23)
    - Le jour de la semaine (0=lundi, 6=dimanche)
    - Si c'est le weekend (samedi ou dimanche)
    - Si c'est la nuit (22h-6h)
    - Si c'est le matin (6h-12h)
    - Si c'est l'après-midi (12h-18h)
    - Si c'est le soir (18h-22h)
    - Si c'est pendant les heures de bureau (8h-17h)
    - L'année, le mois, le jour
    
    Args:
        transaction_time (str or datetime): Heure de la transaction
    
    Returns:
        dict: Dictionnaire avec toutes les features temporelles
        None: Si la date est invalide
    
    Example:
        >>> extract_time_features('2026-01-29 14:30:00')
        {
            'hour': 14,
            'day_of_week': 2,  # Mercredi
            'is_weekend': 0,
            'is_night': 0,
            'is_morning': 0,
            'is_afternoon': 1,
            'is_evening': 0,
            'is_business_hour': 1,
            'year': 2026,
            'month': 1,
            'day': 29
        }
    """
    # Vérifier si la valeur est manquante
    if pd.isna(transaction_time) or transaction_time is None:
        return None
    
    try:
        # Convertir en datetime si nécessaire
        if isinstance(transaction_time, str):
            dt = pd.to_datetime(transaction_time)
        else:
            dt = transaction_time
        
        # Extraire l'heure (0-23)
        hour = dt.hour
        
        # Extraire le jour de la semaine (0=lundi, 6=dimanche)
        day_of_week = dt.dayofweek
        
        # Créer le dictionnaire de features
        features = {
            # Heure brute
            'hour': hour,
            'day_of_week': day_of_week,
            
            # Périodes de la journée (binaire : 0 ou 1)
            'is_night': 1 if 22 <= hour or hour < 6 else 0,        # 22h-6h
            'is_morning': 1 if 6 <= hour < 12 else 0,              # 6h-12h
            'is_afternoon': 1 if 12 <= hour < 18 else 0,           # 12h-18h
            'is_evening': 1 if 18 <= hour < 22 else 0,             # 18h-22h
            'is_business_hour': 1 if 8 <= hour < 17 else 0,        # 8h-17h
            
            # Weekend (samedi=5, dimanche=6)
            'is_we': 1 if day_of_week in [5, 6] else 0,
            
            # Composantes de la date
            'year': dt.year,
            'month': dt.month,
            'day': dt.day
        }
        
        return features
        
    except Exception as e:
        print(f"⚠️ Erreur extraction features temps pour {transaction_time}: {e}")
        return None


# =====================================================================
# FONCTION 4 : FEATURE ENGINEERING COMPLET
# =====================================================================

def engineer_features(transaction_data):
    """
    Applique TOUTES les transformations de features sur une transaction
    
    Cette fonction est la fonction PRINCIPALE qui :
    1. Prend les données brutes d'une transaction
    2. Calcule la distance GPS client-marchand
    3. Extrait les features temporelles
    4. Calcule l'âge si la date de naissance est fournie
    5. Retourne un dictionnaire avec TOUTES les features
    
    Args:
        transaction_data (dict): Dictionnaire avec les données brutes
            Clés requises:
            - lat, long: Coordonnées client
            - merch_lat, merch_long: Coordonnées marchand
            - transaction_time: Heure de transaction
            Clés optionnelles:
            - dob: Date de naissance (pour calculer l'âge)
            - amt, cc_num, etc.: Autres features
    
    Returns:
        dict: Dictionnaire avec toutes les features (brutes + engineered)
    
    Example:
        >>> data = {
        ...     'amt': 150.75,
        ...     'lat': 40.7128,
        ...     'long': -74.0060,
        ...     'merch_lat': 40.7589,
        ...     'merch_long': -73.9851,
        ...     'transaction_time': '2026-01-29 14:30:00',
        ...     'city_pop': 8000000,
        ...     'dob': '1990-01-15'
        ... }
        >>> result = engineer_features(data)
        >>> print(result['distance_km'])
        5.87  # Distance en km
        >>> print(result['hour'])
        14
        >>> print(result['age'])
        36
    """
    # Copier les données pour ne pas modifier l'original
    features = transaction_data.copy()
    
    # ========================================
    # 1. CALCUL DE LA DISTANCE GPS
    # ========================================
    
    # Vérifier que les coordonnées sont présentes
    if all(key in features for key in ['lat', 'long', 'merch_lat', 'merch_long']):
        distance = haversine_distance(
            features['lat'],
            features['long'],
            features['merch_lat'],
            features['merch_long']
        )
        features['distance_km'] = distance
        print(f"  ✅ Distance calculée: {distance} km")
    else:
        features['distance_km'] = None
        print(f"  ⚠️ Coordonnées GPS manquantes, distance non calculée")
    
    # ========================================
    # 2. EXTRACTION DES FEATURES TEMPORELLES
    # ========================================
    
    # Vérifier que l'heure de transaction est présente
    if 'transaction_time' in features:
        time_features = extract_time_features(features['transaction_time'])
        
        if time_features:
            # Ajouter toutes les features temporelles au dictionnaire
            features.update(time_features)
            print(f"  ✅ Features temporelles extraites (heure: {time_features['hour']})")
        else:
            print(f"  ⚠️ Impossible d'extraire les features temporelles")
    else:
        print(f"  ⚠️ Heure de transaction manquante")
    
    # ========================================
    # 3. CALCUL DE L'ÂGE (si date de naissance fournie)
    # ========================================
    
    if 'dob' in features:
        age = calculate_age(features['dob'])
        features['age'] = age
        if age:
            print(f"  ✅ Âge calculé: {age} ans")
        else:
            print(f"  ⚠️ Impossible de calculer l'âge")
    else:
        features['age'] = None
        print(f"  ⚠️ Date de naissance non fournie")
    
    # ========================================
    # RETOUR
    # ========================================
    
    return features


# =====================================================================
# FONCTION 5 : LISTE DES FEATURES POUR LE MODÈLE
# =====================================================================

def get_model_features():
    """
    Retourne la liste EXACTE des features attendues par le modèle ML
    
    Cette fonction définit l'ordre EXACT des colonnes que le modèle attend.
    IMPORTANT: L'ordre DOIT être le même que lors de l'entraînement !
    
    Returns:
        list: Liste des noms de features dans le bon ordre
    
    Categories:
        - Numerical: 17 features numériques
        - Categorical: 4 features catégorielles
    """
    # Features NUMÉRIQUES (17 features)
    numerical_features = [
        'cc_num',           # Numéro de carte (hashé)
        'amt',              # Montant de la transaction
        'zip',              # Code postal
        'city_pop',         # Population de la ville
        'distance_km',      # Distance client-marchand (ENGINEERED)
        'age',              # Âge du porteur (ENGINEERED)
        'hour',             # Heure 0-23 (ENGINEERED)
        'is_night',         # 1 si nuit (ENGINEERED)
        'is_morning',       # 1 si matin (ENGINEERED)
        'is_afternoon',     # 1 si après-midi (ENGINEERED)
        'is_evening',       # 1 si soir (ENGINEERED)
        'is_business_hour', # 1 si heures de bureau (ENGINEERED)
        'year',             # Année (ENGINEERED)
        'month',            # Mois (ENGINEERED)
        'day',              # Jour (ENGINEERED)
        'dayofweek',        # Jour de la semaine (ENGINEERED) - Renommé de 'day_of_week'
        'is_we'             # 1 si weekend (ENGINEERED)
    ]
    
    # Features CATÉGORIELLES (4 features)
    categorical_features = [
        'merchant',         # Nom du marchand
        'category',         # Catégorie de la transaction
        'gender',           # Genre du client
        'state'             # État (US)
    ]
    
    # TOUTES les features dans l'ORDRE
    all_features = numerical_features + categorical_features
    
    return all_features


def prepare_for_model(features_dict):
    """
    Prépare les features dans le bon format pour le modèle
    
    Cette fonction :
    1. Prend le dictionnaire de features
    2. Sélectionne UNIQUEMENT les features nécessaires
    3. Les arrange dans le BON ORDRE
    4. Convertit en DataFrame
    5. Renomme 'day_of_week' en 'dayofweek' (compatibilité modèle)
    
    Args:
        features_dict (dict): Dictionnaire avec toutes les features
    
    Returns:
        pd.DataFrame: DataFrame avec les features dans le bon ordre
        None: Si des features manquent
    
    Example:
        >>> features = engineer_features(transaction_data)
        >>> df_ready = prepare_for_model(features)
        >>> # df_ready est prêt pour model.predict()
    """
    # Renommer 'day_of_week' en 'dayofweek' si présent
    # (Le modèle a été entraîné avec 'dayofweek')
    if 'day_of_week' in features_dict and 'dayofweek' not in features_dict:
        features_dict['dayofweek'] = features_dict['day_of_week']
    
    # Obtenir la liste des features attendues
    expected_features = get_model_features()
    
    # Vérifier que toutes les features sont présentes
    missing_features = [f for f in expected_features if f not in features_dict]
    
    if missing_features:
        print(f"❌ Features manquantes: {missing_features}")
        return None
    
    # Sélectionner uniquement les features nécessaires DANS LE BON ORDRE
    selected_data = {feature: features_dict[feature] for feature in expected_features}
    
    # Convertir en DataFrame (le modèle attend un DataFrame)
    df = pd.DataFrame([selected_data])
    
    print(f"✅ Features préparées: {df.shape[1]} colonnes")
    
    return df


# =====================================================================
# MÉTADONNÉES DU MODULE
# =====================================================================

__version__ = "1.0.0"
__author__ = "Terorra"

# Liste des fonctions exportées
__all__ = [
    'haversine_distance',
    'calculate_age',
    'extract_time_features',
    'engineer_features',
    'get_model_features',
    'prepare_for_model'
]


# =====================================================================
# TEST DU MODULE (si exécuté directement)
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Test du module Feature Engineering")
    print("=" * 70)
    
    # Données de test
    test_transaction = {
        'cc_num': 374125201044065,
        'amt': 150.75,
        'lat': 40.7128,
        'long': -74.0060,
        'city_pop': 8000000,
        'merch_lat': 40.7589,
        'merch_long': -73.9851,
        'transaction_time': '2026-01-29 14:30:00',
        'dob': '1990-01-15',
        'merchant': 'Amazon',
        'category': 'shopping_net',
        'gender': 'M',
        'state': 'NY',
        'zip': 10001
    }
    
    print("\n📊 Données de test:")
    for key, value in test_transaction.items():
        print(f"  {key}: {value}")
    
    print("\n🔧 Application du feature engineering...")
    engineered = engineer_features(test_transaction)
    
    print("\n📊 Features créées:")
    for key in ['distance_km', 'hour', 'is_afternoon', 'age']:
        if key in engineered:
            print(f"  {key}: {engineered[key]}")
    
    print("\n📦 Préparation pour le modèle...")
    df_ready = prepare_for_model(engineered)
    
    if df_ready is not None:
        print(f"✅ Prêt pour prédiction: {df_ready.shape}")
        print(f"   Colonnes: {list(df_ready.columns)}")
    
    print("\n" + "=" * 70)
