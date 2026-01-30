#!/usr/bin/env python3
"""
Initialize Neon DB with 4 tables
-------------------------------
Tables:
1. payments: Raw transaction data from API
2. training_data: Preprocessed data with engineered features
3. predictions: ML predictions
4. fraud_alerts: Fraud notifications

Author: Terorra
Date: January 2026
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def main():
    print("=" * 70)
    print("🚀 Initializing Neon DB for Fraud Detection")
    print("=" * 70)
    
    # Get DATABASE_URL
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ ERROR: DATABASE_URL not found in .env file!")
        print("\nPlease add:")
        print("  DATABASE_URL=postgresql://user:password@host/database")
        sys.exit(1)
    
    # Connect
    print(f"\n📡 Connecting to database...")
    engine = create_engine(db_url)
    
    # Test connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ Connected to PostgreSQL")
            print(f"   Version: {version[:60]}...")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    # Create tables
    print("\n📦 Creating tables...")
    
    sql = """
    -- ================================================================
    -- Table 1: RAW PAYMENTS from API
    -- ================================================================
    CREATE TABLE IF NOT EXISTS payments (
        id SERIAL PRIMARY KEY,
        trans_num VARCHAR(100) UNIQUE NOT NULL,
        cc_num VARCHAR(100),
        merchant VARCHAR(255),
        category VARCHAR(100),
        amt DECIMAL(10, 2),
        first VARCHAR(100),
        last VARCHAR(100),
        gender VARCHAR(10),
        street VARCHAR(255),
        city VARCHAR(100),
        state VARCHAR(10),
        zip VARCHAR(20),
        lat FLOAT,
        long FLOAT,
        city_pop INTEGER,
        job VARCHAR(100),
        dob VARCHAR(20),
        trans_num_original VARCHAR(100),
        merch_lat FLOAT,
        merch_long FLOAT,
        is_fraud INTEGER,
        transaction_time VARCHAR(50),
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- ================================================================
    -- Table 2: TRAINING DATA (preprocessed with features)
    -- ================================================================
    CREATE TABLE IF NOT EXISTS training_data (
        id SERIAL PRIMARY KEY,
        trans_num VARCHAR(100) UNIQUE,
        cc_num VARCHAR(100),
        merchant VARCHAR(255),
        category VARCHAR(100),
        amt DECIMAL(10, 2),
        gender VARCHAR(10),
        state VARCHAR(10),
        zip VARCHAR(20),
        city_pop INTEGER,
        distance_km FLOAT,
        age INTEGER,
        hour INTEGER,
        is_business_hour INTEGER,
        is_morning INTEGER,
        is_afternoon INTEGER,
        is_evening INTEGER,
        is_night INTEGER,
        year INTEGER,
        month INTEGER,
        day INTEGER,
        dayofweek INTEGER,
        is_we INTEGER,
        is_fraud INTEGER,
        created_at TIMESTAMP DEFAULT NOW(),
        FOREIGN KEY (trans_num) REFERENCES payments(trans_num) ON DELETE CASCADE
    );
    
    -- ================================================================
    -- Table 3: PREDICTIONS
    -- ================================================================
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        trans_num VARCHAR(100) UNIQUE,
        is_fraud BOOLEAN NOT NULL,
        fraud_probability FLOAT NOT NULL,
        predicted_at TIMESTAMP DEFAULT NOW(),
        FOREIGN KEY (trans_num) REFERENCES payments(trans_num) ON DELETE CASCADE
    );
    
    -- ================================================================
    -- Table 4: FRAUD ALERTS
    -- ================================================================
    CREATE TABLE IF NOT EXISTS fraud_alerts (
        id SERIAL PRIMARY KEY,
        trans_num VARCHAR(100) UNIQUE,
        email_sent BOOLEAN DEFAULT FALSE,
        email_sent_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW(),
        FOREIGN KEY (trans_num) REFERENCES payments(trans_num) ON DELETE CASCADE
    );
    
    -- ================================================================
    -- INDEXES for performance
    -- ================================================================
    CREATE INDEX IF NOT EXISTS idx_payments_time ON payments(created_at);
    CREATE INDEX IF NOT EXISTS idx_payments_trans_num ON payments(trans_num);
    
    CREATE INDEX IF NOT EXISTS idx_training_time ON training_data(created_at);
    CREATE INDEX IF NOT EXISTS idx_training_trans_num ON training_data(trans_num);
    
    CREATE INDEX IF NOT EXISTS idx_predictions_fraud ON predictions(is_fraud);
    CREATE INDEX IF NOT EXISTS idx_predictions_trans_num ON predictions(trans_num);
    
    CREATE INDEX IF NOT EXISTS idx_alerts_sent ON fraud_alerts(email_sent);
    CREATE INDEX IF NOT EXISTS idx_alerts_trans_num ON fraud_alerts(trans_num);
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
        
        print("✅ Tables created successfully:")
        print("   1. payments - Raw transactions from API")
        print("   2. training_data - Preprocessed with features")
        print("   3. predictions - ML predictions")
        print("   4. fraud_alerts - Fraud notifications")
        print("\n✅ Indexes created for performance")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        sys.exit(1)
    
    # Verify tables
    print("\n🔍 Verifying tables...")
    
    verify_sql = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
    """
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(verify_sql))
            tables = [row[0] for row in result]
            
            print(f"✅ Found {len(tables)} tables:")
            for table in tables:
                print(f"   • {table}")
                
    except Exception as e:
        print(f"⚠️ Verification warning: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Database initialization completed successfully!")
    print("=" * 70)
    
    print("\n📝 Next steps:")
    print("   1. Copy fraud_utils.py to 02_airflow/plugins/")
    print("   2. Copy __init__.py to 02_airflow/plugins/")
    print("   3. Copy DAG to 02_airflow/dags/")
    print("   4. Run: docker compose restart")
    print("   5. Access Airflow: http://localhost:8080")
    
    print("\n💡 Workflow:")
    print("   API → ETL branch → payments table")
    print("        → Preprocess branch → training_data table")
    print("        → Predict branch → predictions + fraud_alerts + email")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
