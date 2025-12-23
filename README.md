# NeuraLearn Microservices API

A comprehensive FastAPI-based microservice for automated data processing, cleaning, exploratory data analysis (EDA), and feature engineering. This service provides intelligent data preprocessing capabilities with AI-powered suggestions using Google's Gemini API.

## 🚀 Features

- **Automated Data Processing Pipeline**: Upload datasets and get them automatically cleaned, analyzed, and feature-engineered
- **AI-Powered Data Cleaning**: Uses Google Gemini API for intelligent missing value handling strategies
- **Comprehensive EDA**: Generates histograms, boxplots, correlation heatmaps, and count plots
- **Smart Feature Engineering**: Automated categorical encoding, feature scaling, and derived feature creation
- **Outlier Detection & Handling**: Multiple methods (IQR, Z-score) with configurable fixing strategies
- **Cloud Storage Integration**: AWS S3 integration for scalable file storage
- **Database Logging**: PostgreSQL database for tracking all processing steps and metadata
- **Collaboration Support**: Multi-user collaboration features with shared datasets
- **RESTful API**: Complete REST API with automatic documentation

## 🏗️ Architecture

```
├── main.py                 # FastAPI application and API endpoints
├── database.py            # Database models and operations (SQLAlchemy)
├── s3_client.py           # AWS S3 integration
├── services/              # Core processing services
│   ├── data_loader.py     # Data loading from files/URLs
│   ├── cleaner.py         # Data cleaning with AI suggestions
│   ├── eda.py             # Exploratory Data Analysis
│   ├── outliers.py        # Outlier detection and handling
│   ├── feature_engineering.py # Feature engineering with AI
│   └── gemini_client.py   # Google Gemini API integration
└── storage/               # Local file storage
    ├── datasets/          # Raw and cleaned datasets
    └── graphs/            # Generated EDA visualizations
```

## 📋 Requirements

- Python 3.8+
- PostgreSQL database
- AWS S3 bucket (optional, for cloud storage)
- Google Gemini API key (optional, for AI features)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd neuralearn-microservices
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Configuration**
Create a `.env` file with the following variables:
```env
# Database
DATABASE_URL=postgresql://username:password@host:port/database

# Google Gemini API (optional)
GEMINI_API_KEY=your_gemini_api_key

# AWS S3 (optional)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
S3_BUCKET_NAME=your_bucket_name
```

5. **Run the application**
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Core Endpoints

#### Dataset Upload
```http
POST /dataset/upload
```
Upload a dataset file or provide a URL for processing.

**Parameters:**
- `file`: CSV/JSON file (multipart/form-data)
- `dataset_url`: URL to dataset (alternative to file)
- `user_id`: User identifier (default: "anonymous")
- `collaboration_id`: Collaboration group ID (optional)

**Response:**
```json
{
  "dataset_id": "uuid",
  "rows": 1000,
  "columns": 10,
  "status": "uploaded"
}
```

#### Get Dataset Status
```http
GET /datasets?user_id={user_id}&collaboration_id={collaboration_id}
```

#### Download Processed Data
```http
GET /dataset/{dataset_id}/raw      # Original dataset
GET /dataset/{dataset_id}/cleaned  # Processed dataset
```

#### EDA Visualizations
```http
GET /dataset/{dataset_id}/eda/graphs           # List all graphs
GET /dataset/{dataset_id}/graph/{graph_id}     # Download specific graph
```

#### Feature Information
```http
GET /dataset/{dataset_id}/features  # Get input/output features
```

#### Processing Logs
```http
GET /dataset/{dataset_id}/logs/cleaning   # Data cleaning logs
GET /dataset/{dataset_id}/logs/outliers   # Outlier handling logs
GET /dataset/{dataset_id}/logs/features   # Feature engineering logs
```

#### Collaboration Features
```http
GET /collaboration/{collaboration_id}/graphs    # All graphs in collaboration
GET /collaboration/{collaboration_id}/datasets/cleaned  # All cleaned datasets
```

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔄 Data Processing Pipeline

1. **Upload**: Dataset uploaded via file or URL
2. **Validation**: Size and format validation (max 100K rows, 100 columns)
3. **Storage**: Raw data saved locally and optionally to S3
4. **Cleaning**: AI-powered missing value handling
5. **EDA**: Automatic generation of visualizations and statistics
6. **Outlier Detection**: IQR/Z-score based outlier detection and fixing
7. **Feature Engineering**: Categorical encoding, scaling, derived features
8. **Completion**: Cleaned dataset and metadata available for download

## 🤖 AI-Powered Features

### Intelligent Data Cleaning
- Analyzes column statistics and data types
- Uses Gemini API to suggest optimal missing value strategies
- Fallback to heuristic-based approaches
- Supports: mean, median, mode, forward-fill, drop column strategies

### Smart Feature Engineering
- Automatic categorical encoding (one-hot, label encoding)
- Feature scaling recommendations (standard, min-max)
- Identifies columns to drop (constant, unique identifiers)
- Suggests derived features based on domain knowledge

## 🗄️ Database Schema

### Tables
- **Dataset**: Main dataset metadata and status
- **CleaningLog**: Data cleaning operations log
- **OutlierLog**: Outlier detection and handling log
- **FeatureLog**: Feature engineering operations log
- **GraphMetadata**: EDA visualization metadata

## ☁️ Cloud Integration

### AWS S3
- Automatic upload of processed datasets
- EDA visualizations storage
- Pre-signed URL generation for secure access
- Configurable retention and cleanup

## 🚀 Deployment

### Heroku Deployment
The project includes a `Procfile` for Heroku deployment:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔧 Configuration

### Data Processing Limits
- Maximum dataset size: 100,000 rows × 100 columns
- Outlier percentage threshold: 10%
- Maximum Gemini API calls per dataset: 3

### Storage Configuration
- Local storage: `storage/` directory
- S3 integration: Optional, configured via environment variables
- File cleanup: Automatic cleanup of temporary files

## 🧪 Testing

Use the included `sample_dataset.csv` for testing:
```bash
curl -X POST "http://localhost:8000/dataset/upload" \
  -F "file=@sample_dataset.csv" \
  -F "user_id=test_user"
```

## 📊 Monitoring & Logging

- Processing status tracking in database
- Comprehensive logging of all operations
- Error handling with detailed error messages
- Background task processing for large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the processing logs via API endpoints
3. Check database logs for detailed error information

---

**Version**: 1.0.0  
**API Documentation**: Available at `/docs` when running