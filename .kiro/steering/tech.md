# NeuraLearn Technical Stack

## Core Framework
- **FastAPI**: Modern, high-performance web framework for building APIs
- **Python 3.8+**: Primary programming language
- **Uvicorn**: ASGI server for production deployment

## Database & Storage
- **PostgreSQL**: Primary database with async support via asyncpg
- **SQLAlchemy 2.0**: ORM with async capabilities and modern mapped_column syntax
- **AWS S3**: Cloud storage for datasets and visualizations with boto3
- **Local Storage**: Fallback storage in `storage/` directory

## AI & Data Processing
- **Google Gemini API**: AI-powered data cleaning and feature engineering suggestions
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning preprocessing (StandardScaler, MinMaxScaler, LabelEncoder)
- **Matplotlib & Seaborn**: Data visualization and EDA graph generation

## Key Dependencies
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
SQLAlchemy[asyncio]==2.0.36
asyncpg==0.30.0
google-generativeai==0.3.2
boto3==1.35.49
scikit-learn==1.3.2
```

## Environment Configuration
Required environment variables in `.env`:
- `DATABASE_URL`: PostgreSQL connection string
- `GEMINI_API_KEY`: Google Gemini API key (optional)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET_NAME`: AWS S3 config (optional)

## Common Commands

### Development
```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload

# Run with specific host/port
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Production Deployment
```bash
# Heroku deployment (uses Procfile)
web: uvicorn main:app --host 0.0.0.0 --port $PORT

# Docker deployment
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Testing
```bash
# Test with sample dataset
curl -X POST "http://localhost:8000/dataset/upload" \
  -F "file=@sample_dataset.csv" \
  -F "user_id=test_user"

# Access API documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

## Architecture Patterns
- **Service Layer Pattern**: Business logic separated into `services/` directory
- **Async/Await**: Full async support for database operations and API endpoints
- **Background Tasks**: FastAPI BackgroundTasks for long-running data processing
- **Dependency Injection**: Database sessions and service instances
- **Error Handling**: Comprehensive exception handling with detailed error messages
- **Logging**: Structured logging for all processing steps stored in database