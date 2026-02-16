"""
DAG Weekly Model Retraining with MLflow
- Runs every Monday at 2 AM
- Trains new model
- Logs to MLflow
- Compares with production model
- Deploys if better performance
- Updates HuggingFace Hub
"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator

from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline

import joblib
import mlflow
import mlflow.sklearn

from huggingface_hub import HfApi, login, hf_hub_download
import os
import numpy as np

default_args = {
    'owner': 'fraud-detection',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(hours=1),
}

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'https://terorra-mlflow-serveur.hf.space')
MLFLOW_EXPERIMENT_NAME = "fraud-detection-retraining"

# Minimum performance thresholds
MIN_RECALL = 0.90
MIN_PRECISION = 0.75
MIN_F1 = 0.80

def setup_mlflow(**context):
    """Configure MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    print(f"✅ MLflow configured:")
    print(f"   Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"   Experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    return True


def extract_training_data(**context):
    """Extract validated transactions for training"""
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not set")
    
    engine = create_engine(database_url)

    query = """
            WITH pred AS (
                SELECT 
                trans_num,
                is_fraud::INTEGER AS is_fraud_pred
            FROM predictions
            ),
            train_d AS (
                SELECT *
                FROM training_data
            )

            SELECT *
            FROM pred pd
            INNER JOIN train_d td 
                ON td.trans_num = pd.trans_num;
        """
    
    df = pd.read_sql(query, engine)
    df = df.drop(columns=['is_fraud', 'created_at'])
    df = df.rename(columns={'is_fraud_pred':'is_fraud'})
    
    print(f"📊 Extracted {len(df)} transactions")
    print(f"   Frauds: {df['is_fraud'].sum()}")
    print(f"   Legit: {(df['is_fraud'] == 0).sum()}")
    print(f"   Fraud rate: {df['is_fraud'].mean():.2%}")
    
    # if len(df) < 1000:
    #     raise ValueError(f"Not enough data: {len(df)} transactions (min 1000)")
    
    # if df['is_fraud'].sum() < 50:
    #     raise ValueError(f"Not enough fraud examples: {df['is_fraud'].sum()} (min 50)")
    
    return df.to_dict('records')


def train_and_log_model(**context):
    """Train new model and log to MLflow"""
    
    ti = context['ti']
    data = ti.xcom_pull(task_ids='extract_data')
    
    if not data:
        raise ValueError("No training data available")
    
    df = pd.DataFrame(data)
    
    # Prepare data
    target = 'is_fraud'

    y = df[target]
    X = df.drop(target, axis=1)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    print(f"🔨 Training new model...")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")

    # load model : 
    hf_repo = os.getenv('HF_MODEL_REPO', 'Terorra/fd_model_jedha')

    print(f"⬇️ Downloading from {hf_repo}...")

    try:
        # Download preprocessor 
        preprocessor_path = hf_hub_download(
            repo_id=hf_repo,
            filename="preprocessor.pkl",
            cache_dir="/tmp"
        )
        
        print(f"✅ Files downloaded")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None
    
    # Load preprocessor
    try:
        preprocessor = joblib.load(preprocessor_path)
        print(f"✅ Preprocessor loaded: {type(preprocessor).__name__}")
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return None

    
    # Start MLflow run
    with mlflow.start_run(run_name=f"retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Get run info early
        run = mlflow.active_run()
        run_id = run.info.run_id
        run_id
        
        # Hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': 1
        }
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("fraud_rate", df['is_fraud'].mean())
        
        # preprocess train test
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"\n📊 Model Performance:")
        print(f"   Recall: {recall:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feat' : preprocessor.get_feature_names_out(),
            'imp': model.feature_importances_
        }).sort_values('imp', ascending=False).head(10)

        print(f"\n📊 Top 10 Feature Importance:")
        print(importance_df)
        
        # ========================================
        # FIX: Log model SANS registered_model_name
        # ========================================
        
        print(f"\n📦 Logging model to MLflow...")
        
        try:
            # Log le modèle (sans registry)
            mlflow.sklearn.log_model(model, name="model")
            
            print(f"✅ Model logged to MLflow")
            # print(f"   Artifact URI: {model_uri}")
            artifact_uri = mlflow.get_artifact_uri("model")
            print(f"   Artifact URI: {artifact_uri}")
            
        except Exception as e:
            print(f"❌ Model logging failed: {e}")
            pass
        
        # ========================================
        # FIX: Register model SÉPARÉMENT
        # ========================================
        
        print(f"\n📝 Registering model...")
        
        try:
            # URI du modèle
            model_uri = f"runs:/{run_id}/model"
            
            # Register le modèle
            model_version = mlflow.register_model(model_uri, "fraud-detector")
            
            print(f"✅ Model registered as 'fraud-detector'")
            print(f"   Version: {model_version.version}")
            
        except Exception as e:
            print(f"⚠️ Model registration failed: {e}")
            print(f"   Model is still logged, but not in registry")
            print(f"   This is OK, you can register manually later")
        
        # Save locally (pour HuggingFace)
        model_path = '/tmp/fraud_model_new.pkl'
        joblib.dump(model, model_path)
        print(f"✅ Model saved locally: {model_path}")
        
        # Log classification report as artifact
        report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])
        
        report_path = '/tmp/classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        try:
            mlflow.log_artifact(report_path)
            print(f"✅ Classification report logged")
        except Exception as e:
            print(f"⚠️ Failed to log report: {e}")
        
        print(f"\n📊 Classification Report:")
        print(report)
        
        # Check if meets requirements
        meets_requirements = (
            recall >= MIN_RECALL and
            precision >= MIN_PRECISION and
            f1 >= MIN_F1
        )
        
        if meets_requirements:
            print(f"\n✅ Model meets requirements!")
        else:
            print(f"\n⚠️ Model does NOT meet requirements:")
            if recall < MIN_RECALL:
                print(f"   Recall: {recall:.2%} < {MIN_RECALL:.2%}")
            if precision < MIN_PRECISION:
                print(f"   Precision: {precision:.2%} < {MIN_PRECISION:.2%}")
            if f1 < MIN_F1:
                print(f"   F1: {f1:.2%} < {MIN_F1:.2%}")
        
        # Return info for downstream tasks
        return {
            'run_id': run_id,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'accuracy': accuracy,
            'meets_requirements': meets_requirements,
            'model_path': model_path
        }


def compare_with_production(**context):
    """Compare new model with production model"""
    
    ti = context['ti']
    new_model_info = ti.xcom_pull(task_ids='train_model')
    
    if not new_model_info:
        raise ValueError("No new model info available")
    
    print(f"\n🔍 Comparing with production model...")
    
    client = mlflow.MlflowClient()
    
    try:
        # Get production model versions
        model_name = "fraud-detector"
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not prod_versions:
            print("ℹ️ No production model found, will deploy new model")
            return {
                'should_deploy': True,
                'reason': 'No production model exists',
                'new_model_info': new_model_info
            }
        
        prod_version = prod_versions[0]
        prod_run_id = prod_version.run_id
        
        # Get production metrics
        prod_run = client.get_run(prod_run_id)
        prod_recall = prod_run.data.metrics.get('recall', 0)
        prod_precision = prod_run.data.metrics.get('precision', 0)
        prod_f1 = prod_run.data.metrics.get('f1_score', 0)
        
        print(f"\n📊 Model Comparison:")
        print(f"\n   Production Model (v{prod_version.version}):")
        print(f"      Recall: {prod_recall:.4f}")
        print(f"      Precision: {prod_precision:.4f}")
        print(f"      F1-Score: {prod_f1:.4f}")
        
        print(f"\n   New Model:")
        print(f"      Recall: {new_model_info['recall']:.4f}")
        print(f"      Precision: {new_model_info['precision']:.4f}")
        print(f"      F1-Score: {new_model_info['f1_score']:.4f}")
        
        # Decision logic
        recall_improvement = new_model_info['recall'] - prod_recall
        precision_change = new_model_info['precision'] - prod_precision
        f1_improvement = new_model_info['f1_score'] - prod_f1
        
        should_deploy = (
            new_model_info['meets_requirements'] and
            recall_improvement > 0.01 and  # At least 1% improvement
            precision_change > -0.05 and   # Max 5% degradation
            f1_improvement > 0
        )
        
        if should_deploy:
            reason = f"Better performance: +{recall_improvement:.2%} recall, +{f1_improvement:.2%} F1"
            print(f"\n✅ DEPLOY: {reason}")
        else:
            reason = f"Not better enough: +{recall_improvement:.2%} recall (need >1%)"
            print(f"\n❌ SKIP: {reason}")
        
        return {
            'should_deploy': should_deploy,
            'reason': reason,
            'prod_recall': prod_recall,
            'prod_precision': prod_precision,
            'prod_f1': prod_f1,
            'new_model_info': new_model_info
        }
        
    except Exception as e:
        print(f"⚠️ Error comparing models: {e}")
        print(f"   Will deploy if new model meets requirements")
        
        return {
            'should_deploy': new_model_info['meets_requirements'],
            'reason': f'Comparison failed: {str(e)}',
            'new_model_info': new_model_info
        }


def deploy_to_production(**context):
    """Deploy model to production (MLflow Registry + HuggingFace)"""
    
    ti = context['ti']
    comparison = ti.xcom_pull(task_ids='compare_models')
    
    if not comparison['should_deploy']:
        print(f"⏭️ Skipping deployment: {comparison['reason']}")
        return {
            'deployed': False,
            'reason': comparison['reason']
        }
    
    new_model_info = comparison['new_model_info']
    run_id = new_model_info['run_id']
    
    print(f"🚀 Deploying model to production...")
    
    # Promote model in MLflow Registry
    client = mlflow.MlflowClient()
    
    try:
        # Get model version
        model_name = "fraud-detector"
        versions = client.search_model_versions(f"run_id='{run_id}'")
        
        if not versions:
            raise ValueError(f"No model version found for run {run_id}")
        
        model_version = versions[0].version
        
        # Transition to Production
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"✅ Model v{model_version} promoted to Production in MLflow")
        
    except Exception as e:
        print(f"⚠️ MLflow promotion failed: {e}")
        model_version = None
    
    # Upload to HuggingFace
    hf_token = os.getenv('HF_TOKEN')
    hf_repo = os.getenv('HF_MODEL_REPO', 'Terorra/fd_model_jedha')
    
    if hf_token:
        try:
            login(token=hf_token)
            api = HfApi()
            
            model_path = new_model_info['model_path']
            
            commit_message = "Retrained model"
            
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo="fraud_model.pkl",
                repo_id=hf_repo,
                repo_type="model",
                commit_message=commit_message
            )
            
            print(f"✅ Model uploaded to HuggingFace: {hf_repo}")
            
        except Exception as e:
            print(f"⚠️ HuggingFace upload failed: {e}")
    else:
        print("ℹ️ HF_TOKEN not set, skipping HuggingFace upload")
    
    return {
        'deployed': True,
        'reason': comparison['reason'],
        'mlflow_version': model_version,
        'metrics': new_model_info
    }


def send_report_email(**context):
    """Send email report about retraining"""
    
    ti = context['ti']
    deployment = ti.xcom_pull(task_ids='deploy_model')
    
    if not deployment:
        print("⚠️ No deployment info")
        return
    
    # Email config
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    alert_email = os.getenv('ALERT_EMAIL_TO')
    
    if not all([smtp_user, smtp_password, alert_email]):
        print("ℹ️ SMTP not configured, skipping email")
        return
    
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    metrics = deployment.get('metrics', {})
    
    status = "✅ DEPLOYED" if deployment['deployed'] else "⏭️ SKIPPED"
    
    msg = MIMEMultipart('alternative')
    msg['From'] = smtp_user
    msg['To'] = alert_email
    msg['Subject'] = f'{status} - Model Retraining Report'
    
    text_content = f"""
MODEL RETRAINING REPORT
{'='*50}

Status: {status}
Reason: {deployment['reason']}

NEW MODEL METRICS
{'-'*50}
Recall: {metrics.get('recall', 0):.2%}
Precision: {metrics.get('precision', 0):.2%}
F1-Score: {metrics.get('f1_score', 0):.2%}
Accuracy: {metrics.get('accuracy', 0):.2%}

{'DEPLOYED TO PRODUCTION' if deployment['deployed'] else 'NOT DEPLOYED'}
{'-'*50}

MLflow: {MLFLOW_TRACKING_URI}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    msg.attach(MIMEText(text_content, 'plain'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        print(f"✅ Report sent to {alert_email}")
    except Exception as e:
        print(f"⚠️ Email failed: {e}")


with DAG(
    dag_id='fraud_model_retraining',
    default_args=default_args,
    description='Weekly model retraining with MLflow tracking',
    schedule='0 2 * * 1',  # Every Monday at 2 AM
    catchup=False,
    tags=['ml', 'retraining', 'mlflow', 'weekly']
) as dag:
    
    start = EmptyOperator(task_id='start')
    
    setup = PythonOperator(
        task_id='setup_mlflow',
        python_callable=setup_mlflow
    )
    
    extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_training_data
    )
    
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_and_log_model
    )
    
    compare = PythonOperator(
        task_id='compare_models',
        python_callable=compare_with_production
    )
    
    deploy = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_to_production
    )
    
    report = PythonOperator(
        task_id='send_report',
        python_callable=send_report_email
    )
    
    end = EmptyOperator(task_id='end')
    
    start >> setup >> extract >> train >> compare >> deploy >> report >> end
