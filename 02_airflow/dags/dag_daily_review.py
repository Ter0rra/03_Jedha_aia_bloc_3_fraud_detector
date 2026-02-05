from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import TaskGroup

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

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


def extract_daily_data(**context):
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
                ON td.trans_num = pd.trans_num
            WHERE td.created_at >= CURRENT_DATE - INTERVAL '1 day'
            AND td.created_at < CURRENT_DATE
            AND pd.is_fraud_pred = 1 ;
        """
    
    df = pd.read_sql(query, engine)
    df = df.drop(columns=['is_fraud'])
    df = df.rename(columns={'is_fraud_pred':'is_fraud'})
    df['created_at'] = df['created_at'].astype('str')
    
    print(f"📊 Extracted {len(df)} transactions")
    print(f'head of df {df.head(5)}')
    print(f' get info {df.info()}')
    print(f"   Frauds: {df['is_fraud'].sum()}")
    print(f"   Legit: {(df['is_fraud'] == 0).sum()}")
    print(f"   Fraud rate: {df['is_fraud'].mean():.2%}")
    
    dict_dr = df.to_dict('records')
    return dict_dr


def get_csv(**context): 
    """Convert data to CSV and save in /tmp"""
    
    ti = context['ti']
    data = ti.xcom_pull(task_ids='daily_report') 
    
    if not data:
        print("⚠️ No data to convert")
        return None
    
    df = pd.DataFrame(data)
    
    # ========================================
    # FIX 1: Utiliser /tmp (existe toujours)
    # FIX 2: Format date SANS slashes (YYYY-MM-DD)
    # ========================================
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')  # 2026-02-04_143025
    filename = f'/tmp/Daily_report_{timestamp}.csv'
    
    print(f"💾 Saving CSV to: {filename}")
    
    # Sauvegarder le CSV
    df.to_csv(filename, index=False)
    
    print(f"✅ CSV saved: {filename}")
    
    # Retourner le chemin pour la tâche suivante
    return filename


def send_report(**context):
    """Send email with CSV attachment"""
    
    ti = context['ti']
    data = ti.xcom_pull(task_ids='daily_report')
    csv_path = ti.xcom_pull(task_ids='trans_to_csv')
    
    if not data:
        print("⚠️ No report for email")
        return 0
    
    if not csv_path:
        print("⚠️ No CSV file to attach")
        return 0
    
    df = pd.DataFrame(data)
    
    # SMTP configuration
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    alert_email = os.getenv('ALERT_EMAIL_TO')
    
    if not all([smtp_user, smtp_password, alert_email]):
        print("❌ SMTP configuration missing")
        return 0
    
    print(f"📧 Sending email to {alert_email}...")
    
    # ========================================
    # FIX 3: Créer email avec pièce jointe
    # ========================================
    
    # Créer message multipart (pour avoir texte + attachement)
    msg = MIMEMultipart()
    msg['Subject'] = f'DAILY REPORT FRAUDE - {len(df)} transaction(s) suspecte(s)'
    msg['From'] = smtp_user
    msg['To'] = alert_email
    
    # Corps du texte
    body = f"""
    DAILY REPORT - {len(df)} transaction(s) suspecte(s) détectée(s)
    {'='*70}

    Statistiques:
    - Total transactions: {len(df)}
    - Frauduleuses: {df['is_fraud'].sum()}
    - Légitimes: {(df['is_fraud'] == 0).sum()}
    - Taux de fraude: {df['is_fraud'].mean():.2%}

    Période: dernières 24 heures
    Fichier CSV en pièce jointe.

    ---
    Fraud Detection System - Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Attacher le CSV
    try:
        with open(csv_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        
        # Nom du fichier dans l'email
        filename = os.path.basename(csv_path)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {filename}'
        )
        
        msg.attach(part)
        
        print(f"✅ CSV attached: {filename}")
        
    except Exception as e:
        print(f"⚠️ Failed to attach CSV: {e}")
        # Continue quand même (envoyer email sans pièce jointe)
    
    # Send email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        print(f"✅ Email sent to {alert_email}")
        
        # Nettoyer le fichier temporaire
        try:
            os.remove(csv_path)
            print(f"🗑️ Temporary file removed: {csv_path}")
        except:
            pass
        
        return len(df)
        
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return 0


# ===========================
# Create DAGs
# ===========================

with DAG(
    dag_id='fraud_daily_report',
    default_args=default_args,
    description='ETL: Neon DB -> csv -> email',
    schedule='0 7 * * *',  # Tous les jours à 7h
    catchup=False,
    tags=['etl', 'daily', 'report']
) as dag:
    
    start = EmptyOperator(task_id='start')
    
    daily_data = PythonOperator(
        task_id='daily_report',
        python_callable=extract_daily_data
    )
    
    transform = PythonOperator(
        task_id='trans_to_csv',
        python_callable=get_csv
    )
    
    send = PythonOperator(
        task_id='report_sending',
        python_callable=send_report
    )
    
    end = EmptyOperator(task_id='end')
    
    start >> daily_data >> transform >> send >> end
