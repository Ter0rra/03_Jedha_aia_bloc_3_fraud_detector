# ============================
# Airflow
# ============================
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import TaskGroup

# ============================
# Test 
# ============================
import pytest

# ============================
# utils
# ============================
from datetime import datetime, timedelta
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from huggingface_hub import hf_hub_download
import os

# =====================================================================
# CONFIGURATION
# =====================================================================

# Arguments par défaut du DAG
default_args = {
    'owner': 'fraud-detection',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 27),  # Lundi 27 janvier 2026
    'retries': 2,              # Retry 2 fois en cas d'échec
    'retry_delay': timedelta(minutes=5),
}


# =====================================================================
# CONFIGURATION
# =====================================================================

api_url = os.getenv('PAYMENT_API_URL')
db_url = os.getenv('DATABASE_URL')
hf_repo = os.getenv('HF_MODEL_REPO', 'Terorra/fd_model_jedha')

# =====================================================================
# Test API
# =====================================================================

def test_api(api=api_url):
    r = requests.get(api)
    assert r.status_code == 200

# =====================================================================
# Test NEONdb
# =====================================================================

def test_neondb_payments(db=db_url):
    engine = create_engine(db)
    df = pd.read_sql_table('payments', engine)
    assert df.empty == False

def test_neondb_predictions(db=db_url):
    engine = create_engine(db)
    df = pd.read_sql_table('predictions', engine)
    assert df.empty == False

def test_neondb_training_data(db=db_url):
    engine = create_engine(db)
    df = pd.read_sql_table('training_data', engine)
    assert df.empty == False

def test_neondb_fraud_alerts(db=db_url):
    engine = create_engine(db_url)
    df = pd.read_sql_table('fraud_alerts', engine)
    assert df.empty == False

# =====================================================================
# Test HF_model
# =====================================================================

def test_hf_model_model(repo=hf_repo):
    try :
        model_path = hf_hub_download(
            repo_id=repo,
            filename="fraud_model.pkl"
        )
        assert model_path
    except Exception as e:
        print(f"❌ Download failed: {e}")

def test_hf_model_preprocessor(repo=hf_repo):
    try:
        preprocessor_path = hf_hub_download(
            repo_id=hf_repo,
            filename="preprocessor.pkl"
        )
        assert preprocessor_path
    except Exception as e:
        print(f"❌ Download failed: {e}")


# =====================================================================
# DAG DEFINITION
# =====================================================================

with DAG(
    dag_id='weekly_tests',
    default_args=default_args,
    description='Tests hebdomadaires complets du projet Fraud Detection',
    schedule='0 9 * * 1',  # Cron: Tous les lundis à 9h00
    catchup=False,
    tags=['tests', 'ETL_pred', 'weekly'],
    doc_md=__doc__
) as dag:
    
    # =================================================================
    # TÂCHE START
    # =================================================================
    
    start = EmptyOperator(
        task_id='start',
        doc_md="""
        ## Début des Tests Hebdomadaires
        
        Ce DAG lance tous les tests de qualité du projet.
        """
    )
    
    # =================================================================
    # TÂCHE API
    # =================================================================
   
    Test_api_call = PythonOperator(
                task_id='call_api',
                python_callable=test_api
            )
    
    # =================================================================
    # TÂCHE HF_model
    # =================================================================

    with TaskGroup(group_id="HF_model") as HF_branch:
        model_load = PythonOperator(
            task_id='model_load',
            python_callable=test_hf_model_model
        )

        prepro_load = PythonOperator(
            task_id='prepro_load',
            python_callable=test_hf_model_preprocessor
        )

        prepro_load >> model_load

    # =================================================================
    # TÂCHE Neondb
    # =================================================================

    with TaskGroup(group_id="Neondb") as NDB_branch:
        payments_conn = PythonOperator(
            task_id='payments_conn',
            python_callable=test_neondb_payments
        )

        predictions_conn = PythonOperator(
            task_id='predictions_conn',
            python_callable=test_neondb_predictions
        )

        training_data_conn = PythonOperator(
            task_id='training_data_conn',
            python_callable=test_neondb_training_data
        )

        fraud_alerts_conn = PythonOperator(
            task_id='fraud_alerts_conn',
            python_callable=test_neondb_fraud_alerts
        )

        payments_conn >> predictions_conn >> training_data_conn >> fraud_alerts_conn


    # =================================================================
    # TÂCHE END
    # =================================================================
    
    end = EmptyOperator(
        task_id='end',
        trigger_rule='all_done',  # S'exécute même si certaines tâches échouent
        doc_md="""
        ## Fin des Tests
        
        Tous les tests sont terminés.
        """
    )
    
    # =================================================================
    # DÉFINITION DU WORKFLOW
    # =================================================================
    
    # Workflow:
    start >> Test_api_call >> HF_branch >> NDB_branch >> end
