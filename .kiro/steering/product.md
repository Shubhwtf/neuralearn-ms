# NeuraLearn Product Overview

NeuraLearn is an intelligent data processing microservice that automates the entire data preprocessing pipeline for machine learning projects. It combines traditional data science techniques with AI-powered decision making using Google's Gemini API.

## Core Value Proposition
- **Automated Data Processing**: Upload datasets and get them automatically cleaned, analyzed, and feature-engineered
- **AI-Powered Intelligence**: Smart suggestions for handling missing values, outlier detection, and feature engineering
- **Comprehensive Analysis**: Automatic generation of EDA visualizations and statistical summaries
- **Collaboration Ready**: Multi-user support with shared datasets and analyses
- **Production Ready**: Cloud storage integration, database logging, and RESTful API

## Key Features
- Data cleaning with intelligent missing value handling
- Outlier detection and correction (IQR and Z-score methods)
- Automated feature engineering (encoding, scaling, derived features)
- EDA visualization generation (histograms, boxplots, correlation heatmaps, count plots)
- AWS S3 integration for scalable storage
- PostgreSQL database for comprehensive logging and metadata
- Background processing for large datasets
- Collaboration features for team workflows

## Target Users
- Data scientists needing automated preprocessing
- ML engineers building data pipelines
- Teams requiring collaborative data analysis
- Organizations wanting standardized data processing workflows

## Processing Limits
- Maximum dataset size: 100,000 rows × 100 columns
- Outlier percentage threshold: 10%
- Maximum Gemini API calls per dataset: 3
- File formats supported: CSV, JSON