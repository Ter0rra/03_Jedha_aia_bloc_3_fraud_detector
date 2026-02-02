---
title: Fraud Detection Dashboard
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
license: mit
---

# 🚨 Fraud Detection Dashboard

Real-time credit card fraud detection monitoring dashboard powered by Machine Learning.

## Features

- 📊 **Real-time KPIs**: Total transactions, fraud rate, average amounts
- 📈 **Analytics Charts**: Time series, distributions, geographic analysis
- 📋 **Transaction Table**: Recent predictions with fraud probability
- 🚨 **Fraud Alerts**: Monitor detected frauds and email notifications
- 🔄 **Auto-refresh**: Data updates every 30 seconds

## Technology Stack

- **Frontend**: Streamlit
- **Database**: Neon DB (PostgreSQL)
- **Visualization**: Plotly
- **ML Model**: RandomForest (hosted on HuggingFace)
- **Deployment**: HuggingFace Spaces (Docker)

## Configuration

Set the `DATABASE_URL` environment variable in HuggingFace Spaces settings:

```
DATABASE_URL=postgresql://user:password@host/database
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export DATABASE_URL="your_neon_db_url"

# Run locally
streamlit run streamlit_app.py
```

## Data Sources

- **Payments**: Real-time transaction data
- **Predictions**: ML model fraud predictions
- **Alerts**: Fraud detection alerts

## License

MIT


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
