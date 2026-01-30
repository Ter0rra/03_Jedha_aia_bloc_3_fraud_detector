from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import TaskGroup

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from datetime import datetime, timedelta, date
import requests
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sqlalchemy import create_engine, text
from huggingface_hub import hf_hub_download
import joblib
import os

# ===========================
# default arg
# ===========================

default_args = {
    'owner': 'fraud-detection',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# ===========================
# make some custom functtion 
# ===========================

# => support function 
# --------------------------------------------------

def age(born):
    """
    Calculate age from birth date
    
    Args:
        born (str): Birth date in format 'YYYY-MM-DD'
    
    Returns:
        int: Age in years, or None if invalid
    
    Example:
        >>> age('1990-01-15')
        36
    """
    if pd.isna(born) or born is None or born == '':
        return None
    
    try:
        born_date = datetime.strptime(str(born), '%Y-%m-%d').date()
        today = date.today()
        return today.year - born_date.year - ((today.month, today.day) < (born_date.month, born_date.day))
    except Exception as e:
        print(f"⚠️ Error calculating age for {born}: {e}")
        return None

def distance_cus_mer(lat1, lon1, lat2, lon2):
            
    """
    Calculate distance between two GPS coordinates using Haversine formula
    Fallback if geopy is not available
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        float: Distance in kilometers
    """
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return None
    
    try:
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return round(R * c, 2)
    except Exception as e:
        print(f"⚠️ Error calculating haversine distance: {e}")
        return None
    
def make_date_feature(df, col='transaction_time'):
    """
    Add comprehensive time-based features
    
    Features added:
    - time: Time of day
    - hour: Hour (0-23)
    - is_night: 22h-6h
    - is_morning: 6h-12h
    - is_afternoon: 12h-18h
    - is_evening: 18h-22h
    - is_business_hour: 8h-17h
    - year, month, day
    - dayofweek: 0=Monday, 6=Sunday
    - is_we: Weekend flag
    
    Args:
        df (DataFrame): Input data
        col (str): Name of datetime column
    
    Returns:
        None (modifies df in-place)
    """
    if col not in df.columns:
        print(f"⚠️ Column '{col}' not found for date features")
        return
    
    try:
        # Parse datetime
        df[col] = pd.to_datetime(df[col])
        
        # Time features
        df['time'] = df[col].dt.time
        df['hour'] = df[col].dt.hour
        
        # Time periods
        df['is_night'] = df['hour'].between(22, 6, inclusive="left").astype(int)
        df['is_morning'] = df['hour'].between(6, 12, inclusive="left").astype(int)
        df['is_afternoon'] = df['hour'].between(12, 18, inclusive="left").astype(int)
        df['is_evening'] = df['hour'].between(18, 22, inclusive="left").astype(int)
        df['is_business_hour'] = df['hour'].between(8, 17).astype(int)
        
        # Date components
        df['year'] = df[col].dt.year
        df['month'] = df[col].dt.month
        df['day'] = df[col].dt.day
        df['dayofweek'] = df[col].dt.day_of_week
        df['is_we'] = df['dayofweek'].between(5, 6).astype(int)
        
        print(f"  ✅ Time features added from '{col}'")
        
    except Exception as e:
        print(f"  ⚠️ Time features failed: {e}")


# => apply function 
# --------------------------------------------------

def add_age_feature(df, dob_column='dob', verbose=True):
    """
    Add age feature to DataFrame
    
    Args:
        df (DataFrame): Input data
        dob_column (str): Name of date of birth column
        verbose (bool): Print progress
    
    Returns:
        DataFrame: Data with 'age' column added
    """
    df = df.copy()
    
    if dob_column not in df.columns:
        if verbose:
            print(f"  ⚠️ Column '{dob_column}' not found")
        return df
    
    try:
        df['age'] = df[dob_column].apply(age)
        valid_ages = df['age'].notna().sum()
        if verbose:
            print(f"  ✅ Age feature added ({valid_ages}/{len(df)} valid)")
        return df
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Age feature failed: {e}")
        return df

def add_distance_feature(df, 
                         client_lat='lat', client_lon='long',
                         merchant_lat='merch_lat', merchant_lon='merch_long',
                         verbose=True):
    """
    Add distance feature between client and merchant using geopy
    
    Args:
        df (DataFrame): Input data
        client_lat, client_lon: Client coordinate columns
        merchant_lat, merchant_lon: Merchant coordinate columns
        verbose (bool): Print progress
    
    Returns:
        DataFrame: Data with 'distance_km' column added
    """
    df = df.copy()
    
    required_cols = [client_lat, client_lon, merchant_lat, merchant_lon]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        if verbose:
            print(f"  ⚠️ Missing columns for distance: {missing_cols}")
        return df
    
    try:
        df['distance_km'] = df.apply(
            lambda row: distance_cus_mer(
                row[client_lat], row[client_lon],
                row[merchant_lat], row[merchant_lon]
            ),
            axis=1
        )
        
        valid_distances = df['distance_km'].notna().sum()
        if verbose:
            print(f"  ✅ Distance feature added ({valid_distances}/{len(df)} valid)")
        return df
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Distance feature failed: {e}")
        return df
    
def add_time_features(df, time_column='transaction_time', verbose=True):
    """
    Add comprehensive time-based features
    
    Args:
        df (DataFrame): Input data
        time_column (str): Name of datetime column
        verbose (bool): Print progress
    
    Returns:
        DataFrame: Data with time features added
    """
    df = df.copy()
    
    if time_column not in df.columns:
        if verbose:
            print(f"  ⚠️ Column '{time_column}' not found")
        return df
    
    try:
        make_date_feature(df, col=time_column)
        
        if verbose:
            features = ['hour', 'is_night', 'is_morning', 'is_afternoon', 
                       'is_evening', 'is_business_hour', 'dayofweek', 'is_we']
            print(f"  ✅ Time features added: {features}")
        return df
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Time features failed: {e}")
        return df

def add_engineered_features(df, verbose=True):
    """
    Add ALL engineered features to DataFrame
    Main function used in DAG
    
    Features added:
    - age: Age from dob
    - distance_km: Customer-merchant distance
    - hour, is_night, is_morning, is_afternoon, is_evening, is_business_hour
    - year, month, day, dayofweek, is_we
    
    Args:
        df (DataFrame): Input data
        verbose (bool): Print progress messages
    
    Returns:
        DataFrame: Data with all engineered features
    """
    df = df.copy()
    
    if verbose:
        print(f"🔧 Feature Engineering (starting with {len(df)} rows):")
    
    # Add age
    df = add_age_feature(df, verbose=verbose)
    
    # Add distance (using geopy)
    df = add_distance_feature(df, verbose=verbose)
    
    # Add time features
    df = add_time_features(df, verbose=verbose)
    
    # Note: amt_per_capita not added by default (uncomment if needed)
    # df = add_amount_features(df, verbose=verbose)
    
    if verbose:
        possible_features = ['age', 'distance_km', 'hour', 'is_night', 'is_morning',
                            'is_afternoon', 'is_evening', 'is_business_hour',
                            'year', 'month', 'day', 'dayofweek', 'is_we']
        added_features = [f for f in possible_features if f in df.columns]
        print(f"✅ Feature engineering complete - Added: {len(added_features)} features")
    
    return df

def drop_columns_for_training(df, verbose=True):
    """
    Drop columns not needed for ML training
    Keeps only features used by the model
    
    Columns to drop:
    - lat, long, merch_lat, merch_long (used for distance calculation)
    - first, last, job, dob (personal info)
    - trans_date_trans_time, unix_time (raw time)
    - city, street (location details)
    - time (redundant with hour)
    - trans_num (ID, not a feature)
    
    Args:
        df (DataFrame): Data with all columns
        verbose (bool): Print progress
    
    Returns:
        DataFrame: Data with only training features
    """
    df = df.copy()
    
    # Columns to drop
    cols_to_drop = ['lat', 'long', 'merch_lat', 'merch_long',
                    'first', 'last', 'job', 'dob',
                    'transaction_time',
                    'city', 'street', 'time']
    
    # Only drop columns that exist
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        if verbose:
            print(f"  ✅ Dropped {len(existing_cols_to_drop)} columns: {existing_cols_to_drop}")
    
    return df


# ===========================
# Extract payment info
# ===========================

def extract_from_api(**context):
    """EXTRACT: Fetch from external API"""
    api_url = os.getenv('PAYMENT_API_URL')
    
    print(f"📥 Fetching from API: {api_url}")
    
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Fetched {len(data)} transactions")
        
        df = pd.read_json(data, orient='split')
        
        # Rename current_time to transaction_time
        if 'current_time' in df.columns:
            df['current_time'] = df['current_time'].astype(str)
            df = df.rename(columns={'current_time': 'transaction_time'})
            df['merchant'] = df['merchant'].str.split('fraud_').str[-1]
        
        dict_api = df.to_dict('records')
        return dict_api
        
    except Exception as e:
        print(f"❌ Error fetching from API: {e}")
        return []
    
# ===========================
# Transform
# ===========================

def transform_raw(**context):
    """TRANSFORM: Single transaction mode"""
    ti = context['ti']
    dict_api = ti.xcom_pull(task_ids='EL_data.api_load')

    if not dict_api:
        print("⚠️ No data to transform")
        return None

    try:
        df_api = pd.DataFrame(dict_api)
        
        print(f"📊 Processing {len(df_api)} transaction(s)")
        return df_api.to_dict('records')
        
    except Exception as e:
        print(f"❌ Transform error: {e}")
        return None

def transform_data_predict(**context):
    """TRANSFORM: Clean and prepare data raw"""
    ti = context['ti']
    dict_api = ti.xcom_pull(task_ids='EL_data.api_load')

    if not dict_api:
        print("⚠️ No data to transform")
        return None

    try:
        df_api2 = pd.DataFrame(dict_api)
        
        # Basic cleaning
        print(f"📊 Original shape: {df_api2.shape}")
        
        # # Drop duplicates
        # if 'trans_num' in df_api2.columns:
        #     df_api2 = df_api2.drop_duplicates(subset=['trans_num'])
        #     print(f"After dedup: {df_api2.shape}")

        df_api2 = add_engineered_features(df_api2)
        df_api2 = drop_columns_for_training(df_api2)

        dict_api3 = df_api2.to_dict('records')
        return dict_api3
        
    except Exception as e:
        print(f"⚠️ Transform error: {e}")
        return None

# ===========================
# load to Neon
# ===========================

def load_to_db(**context): 
    """LOAD: Insert to Neon DB"""
    ti = context['ti']
    dict_api2 = ti.xcom_pull(task_ids='EL_data.transform') 

    if not dict_api2:
        print("⚠️ No data to load")
        return 0

    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ DATABASE_URL not set")
        return 0

    engine = create_engine(db_url)

    try:
        df_api2 = pd.DataFrame(dict_api2)
        
        # Insert row by row to handle duplicates
        inserted = 0
        failed = 0
        
        for _, row in df_api2.iterrows():
            try:
                row_df = pd.DataFrame([row])
                row_df.to_sql('payments', engine, if_exists='append', index=False)
                inserted += 1
            except Exception as e:
                # Likely duplicate
                failed += 1
        
        print(f"✅ Loaded {inserted} transactions to Neon DB")
        if failed > 0:
            print(f"⚠️ Skipped {failed} duplicates")
        
        return inserted
        
    except Exception as e:
        print(f"❌ Load error: {e}")
        return 0
    
def load_to_db_preprocess(**context): 
    """LOAD: Insert to Neon DB"""
    ti = context['ti']
    dict_api2 = ti.xcom_pull(task_ids='TL_prepo.transform_prepro') 

    if not dict_api2:
        print("⚠️ No data to load")
        return 0

    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ DATABASE_URL not set")
        return 0

    engine = create_engine(db_url)

    try:
        df_api2 = pd.DataFrame(dict_api2)

        # # Insert row by row to handle duplicates
        inserted = 0
        failed = 0
        
        for _, row in df_api2.iterrows():
            try:
                row_df = pd.DataFrame([row])
                row_df.to_sql('training_data', engine, if_exists='append', index=False)
                inserted += 1
            except Exception as e:
                # Likely duplicate
                failed += 1
        
        print(f"✅ Loaded {inserted} transactions to Neon DB")
        if failed > 0:
            print(f"⚠️ Skipped {failed} duplicates")
        
        return inserted
        
    except Exception as e:
        print(f"❌ Load error: {e}")
        return 0
    




# =============================
# Predict, load and send email
# =============================

def predict_with_hf_model(**context):
    """Predict using model from HuggingFace"""
    
    ti = context['ti']
    data = ti.xcom_pull(task_ids='TL_prepo.transform_prepro')
    
    if not data:
        print("⚠️ No data to predict")
        return None
    
    df = pd.DataFrame(data)
    
    # Download model from HuggingFace
    hf_repo = os.getenv('HF_MODEL_REPO', 'Terorra/fd_model_jedha')
    
    print(f"⬇️ Downloading from {hf_repo}...")
    
    # https://huggingface.co/Terorra/fd_model_jedha/resolve/main/preprocessor.plk
    # https://huggingface.co/Terorra/fd_model_jedha/resolve/main/fraud_model.pkl

    try:
        # Download preprocessor 
        preprocessor_path = hf_hub_download(
            repo_id=hf_repo,
            filename="preprocessor.pkl",
            cache_dir="/tmp"
        )
        
        # Download model
        model_path = hf_hub_download(
            repo_id=hf_repo,
            filename="fraud_model.pkl",
            cache_dir="/tmp"
        )
        
        print(f"✅ Files downloaded")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None
    
    # Load preprocessor and model
    try:
        preprocessor = joblib.load(preprocessor_path)
        print(f"✅ Preprocessor loaded: {type(preprocessor).__name__}")
        
        model = joblib.load(model_path)
        print(f"✅ Model loaded: {type(model).__name__}")
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return None

    # Prepare features    
    numerical_feat = ['cc_num', 'amt', 'zip', 'city_pop', 'distance_km', 'age', 'hour', 'is_night', 'is_morning', 'is_afternoon', 'is_evening', 'is_business_hour', 'year', 'month', 'day', 'dayofweek', 'is_we']
    categorical_feat = ['merchant', 'category', 'gender', 'state']
    
    # Check if all features exist
    missing_num_features = [col for col in numerical_feat if col not in df.columns]
    if missing_num_features:
        print(f"❌ Missing features: {missing_num_features}")
        return None
    
    missing_cat_features = [col for col in categorical_feat if col not in df.columns]
    if missing_cat_features:
        print(f"❌ Missing features: {missing_cat_features}")
        return None
    
    all_features = numerical_feat + categorical_feat
    
    X = df[all_features]
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [feature for feature in X.columns if feature not in numerical_features]
    
    # Preprocess 
    try:
        X_transformed = preprocessor.transform(X)
        print(f"✅ Data preprocessed: {X_transformed.shape}")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None    

    # Predict
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)[:, 1]
    
    # Add predictions to dataframe
    df['is_fraud_pred'] = predictions
    df['fraud_probability'] = probabilities
    
    fraud_count = predictions.sum()
    print(f"🚨 Detected {fraud_count} potential frauds out of {len(df)} transactions")
    
    return df.to_dict('records')


def save_predictions(**context):
    """Save predictions to Neon DB"""
    
    ti = context['ti']
    data = ti.xcom_pull(task_ids='predict_etls.predict') 
    
    if not data:
        print("⚠️ No predictions to save")
        return 0
    
    df = pd.DataFrame(data)
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not set")
        return 0
        
    engine = create_engine(database_url)
    
    # Prepare predictions table
    pred_df = df[['trans_num', 'is_fraud_pred', 'fraud_probability']].copy()
    pred_df.columns = ['trans_num', 'is_fraud', 'fraud_probability']
    pred_df['is_fraud'] = pred_df['is_fraud'].astype(bool)
    pred_df['predicted_at'] = datetime.utcnow()
    
    # Insert predictions
    try:
        pred_df.to_sql('predictions', engine, if_exists='append', index=False)
        print(f"✅ Saved {len(pred_df)} predictions")
    except Exception as e:
        print(f"❌ Failed to save predictions: {e}")
        return 0
    
    # Insert fraud alerts for fraudulent transactions
    frauds = df[df['is_fraud_pred'] == 1][['trans_num']].copy()
    
    if not frauds.empty:
        try:
            frauds.to_sql('fraud_alerts', engine, if_exists='append', index=False)
            print(f"🚨 Created {len(frauds)} fraud alerts")
        except Exception as e:
            print(f"⚠️ Failed to save fraud alerts: {e}")
    
    return len(pred_df)


def send_fraud_email(**context):
    """Send simple email notification for frauds"""
    
    ti = context['ti']
    data = ti.xcom_pull(task_ids='predict_etls.predict')
    
    if not data:
        print("⚠️ No data for email check")
        return 0
    
    # Filter frauds (probability > 50%)
    df = pd.DataFrame(data)
    frauds = df[df['fraud_probability'] > 0.5].copy()
    
    if frauds.empty:
        print("✅ No frauds to email")
        return 0
 
    print(f"🚨 Found {len(frauds)} frauds to report")
    
    # SMTP configuration
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    alert_email = os.getenv('ALERT_EMAIL_TO')
    
    if not all([smtp_user, smtp_password, alert_email]):
        print("❌ SMTP configuration missing")
        return 0
    
    print(f"📧 Sending email to {alert_email}...")
    
    # Prepare simple text email
    msg = MIMEText(f"""
            ALERTE FRAUDE - {len(frauds)} transaction(s) suspecte(s) détectée(s)
            {'='*70}

            {len(frauds)} transaction(s) frauduleuse(s) détectée(s) par le système ML

            DÉTAILS :
            {'-'*70}

            """ + "\n".join([
            f"""
            Transaction #{i+1}
            ID: {row['trans_num'][:30]}
            Montant: ${row['amt']:.2f}
            Marchand: {row.get('merchant', 'N/A')[:40]}
            Ville: {row.get('city', 'N/A')}, {row.get('state', 'N/A')}
            Probabilité fraude: {row['fraud_probability']:.1%}
            {'-'*70}
            """ for i, (_, row) in enumerate(frauds.head(10).iterrows())
            ]) + f"""

            {f"... et {len(frauds) - 10} autre(s) fraude(s)" if len(frauds) > 10 else ""}

            ACTIONS RECOMMANDÉES :
            - Vérifier ces transactions dans le dashboard
            - Contacter les clients si nécessaire
            - Bloquer les cartes suspectes

            ---
            Fraud Detection System - Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}
            """)
    
    msg['Subject'] = f'🚨 FRAUDE - {len(frauds)} transaction(s) suspecte(s)'
    msg['From'] = smtp_user
    msg['To'] = alert_email
    
    # Send email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        print(f"✅ Email sent to {alert_email}")
        return len(frauds)
        
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return 0


# ===========================
# Create DAGs
# ===========================

with DAG(
    dag_id='fraud_detect',
    default_args=default_args,
    description='ETL: API → Neon DB + ML Predictions',
    schedule=timedelta(seconds=20),
    catchup=False,
    tags=['etl', 'ml', 'unified']
) as dag:
    
    start = EmptyOperator(task_id='start')
    
    with TaskGroup(group_id="EL_data") as load_branch:
        Extract_api = PythonOperator(
            task_id='api_load',
            python_callable=extract_from_api
        )
        
        transform = PythonOperator(
                task_id='transform',
                python_callable=transform_raw
            )
        
        Extract_api >> transform

    load = PythonOperator(
            task_id='load_db',
            python_callable=load_to_db
        )

    with TaskGroup(group_id="TL_prepo") as prepo_branch:

        transform_prepro = PythonOperator(
                task_id='transform_prepro',
                python_callable=transform_data_predict
            )

        load_transform = PythonOperator(
                task_id='load_prepro_db',
                python_callable=load_to_db_preprocess
            )

        transform_prepro >> load_transform

    with TaskGroup(group_id="predict_etls") as predict_branch:
        predict = PythonOperator(
            task_id='predict',
            python_callable=predict_with_hf_model
        )

        load_pred = PythonOperator(
            task_id='save_predict',
            python_callable=save_predictions
        )

        email = PythonOperator(
            task_id='send_fraud_email',
            python_callable=send_fraud_email
        )

        predict >> [load_pred , email]
    
    end = EmptyOperator(task_id='end')
    
    start >> load_branch >> load >> prepo_branch >> predict_branch >> end