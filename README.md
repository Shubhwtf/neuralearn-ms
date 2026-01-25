# NeuraLearn Microservices API

A comprehensive FastAPI-based microservice for automated data processing, cleaning, exploratory data analysis (EDA), and feature engineering. This service provides intelligent data preprocessing capabilities with AI-powered suggestions using Google's Gemini API.

## 🚀 Features

- **Automated Data Processing Pipeline**: Upload datasets and get them automatically cleaned, analyzed, and feature-engineered
- **Multiple Cleaning Modes**: Choose from fast (basic), smart (limited AI), or deep (comprehensive AI) cleaning modes
- **AI-Powered Data Cleaning**: Uses Google Gemini API for intelligent missing value handling strategies
- **Detailed Cleaning Reports**: Deep mode provides comprehensive reports with reasoning, examples, and recommendations
- **Comprehensive EDA**: Generates histograms, boxplots, correlation heatmaps, and count plots
- **Smart Feature Engineering**: Automated categorical encoding, feature scaling, and derived feature creation
- **Outlier Detection & Handling**: Multiple methods (IQR, Z-score) with configurable fixing strategies
- **Cloud Storage Integration**: AWS S3 integration for scalable file storage with Signature Version 4 support
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
│   ├── gemini_client.py   # Google Gemini API integration
│   └── schema.py          # Pydantic models for API responses
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
- `mode`: Cleaning mode - `"fast"` (default), `"smart"`, or `"deep"` (optional)

**Cleaning Modes:**
- **fast**: Basic cleaning without AI, unlimited usage, quick processing
- **smart**: Limited AI assistance (first 10 rows for context), max 3 calls per user
- **deep**: Full AI reasoning with detailed report, max 1 call per user

**Response:**
```json
{
  "dataset_id": "uuid",
  "rows": 1000,
  "columns": 10,
  "status": "uploaded",
  "mode": "fast"
}
```

#### Get Dataset Status
```http
GET /dataset/{dataset_id}/status   # Single dataset status
GET /datasets/status?user_id={user_id}&collaboration_id={collaboration_id}  # Multiple datasets
GET /datasets?user_id={user_id}&collaboration_id={collaboration_id}  # List datasets
```

**Status Response includes:**
- `database_name`: Extracted from filename (e.g., "x" from "x.csv")
- `mode`: Cleaning mode used
- `progress_info`: Human-readable progress information

#### Download Processed Data
```http
GET /dataset/{dataset_id}/raw      # Original dataset
GET /dataset/{dataset_id}/cleaned  # Processed dataset
```

#### Cleaning Report (Deep Mode Only)
```http
GET /dataset/{dataset_id}/report
```
Get detailed cleaning report for datasets processed with `deep` mode.

**Response:**
```json
{
  "dataset_id": "uuid",
  "mode": "deep",
  "reasoning": "Detailed explanation with examples...",
  "summary": "Executive summary of data quality...",
  "recommendations": "Actionable recommendations...",
  "created_at": "2024-01-25T20:17:32.376294"
}
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
GET /collaboration/{collaboration_id}/datasets/cleaned  # All cleaned datasets (includes database_name)
```

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔄 Data Processing Pipeline

1. **Upload**: Dataset uploaded via file or URL with selected cleaning mode
2. **Validation**: Size and format validation (max 100K rows, 100 columns)
3. **Storage**: Raw data saved locally and optionally to S3
4. **Cleaning**: Mode-specific missing value handling
   - **Fast**: Heuristic-based cleaning (mean, median, mode, forward-fill)
   - **Smart**: Limited AI assistance for columns with >20% nulls (up to 3 Gemini calls)
   - **Deep**: Full AI reasoning for top 3 columns with >10% nulls (1 Gemini call with detailed report)
5. **EDA**: Automatic generation of visualizations and statistics
6. **Outlier Detection**: IQR/Z-score based outlier detection and fixing
7. **Feature Engineering**: Categorical encoding, scaling, derived features
8. **Report Generation**: Deep mode generates comprehensive cleaning report
9. **Completion**: Cleaned dataset and metadata available for download

## 🤖 AI-Powered Features

### Intelligent Data Cleaning Modes

#### Fast Mode (Default)
- **No AI usage**: Pure heuristic-based cleaning
- **Unlimited usage**: No rate limits
- **Quick processing**: Fastest option
- **Strategies**: Mean, median, mode, forward-fill, drop column based on data type

#### Smart Mode
- **Limited AI**: Uses Gemini API for columns with >20% nulls
- **Context**: First 10 rows used for AI context
- **Rate limit**: Maximum 3 calls per user
- **Balanced**: Good mix of speed and intelligence

#### Deep Mode
- **Full AI reasoning**: Comprehensive Gemini analysis
- **Detailed report**: Includes reasoning, examples, and recommendations
- **Top columns**: Focuses on top 3 columns with >10% nulls
- **Rate limit**: Maximum 1 call per user
- **Best quality**: Most thorough cleaning with detailed documentation

### Smart Feature Engineering
- Automatic categorical encoding (one-hot, label encoding)
- Feature scaling recommendations (standard, min-max)
- Identifies columns to drop (constant, unique identifiers)
- Suggests derived features based on domain knowledge

## 🗄️ Database Schema

### Tables
- **Dataset**: Main dataset metadata, status, mode, and database_name
- **CleaningLog**: Data cleaning operations log
- **CleaningReport**: Detailed cleaning reports for deep mode (reasoning, summary, recommendations)
- **UserModeUsage**: Rate limiting tracking for smart and deep modes
- **OutlierLog**: Outlier detection and handling log
- **FeatureLog**: Feature engineering operations log
- **GraphMetadata**: EDA visualization metadata

## ☁️ Cloud Integration

### AWS S3
- Automatic upload of processed datasets
- EDA visualizations storage
- Pre-signed URL generation for secure access
- Configurable retention and cleanup
- **Signature Version 4**: Explicitly configured for compatibility with all AWS regions

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
- **Cleaning Mode Limits:**
  - Fast mode: Unlimited
  - Smart mode: 3 calls per user
  - Deep mode: 1 call per user

### Storage Configuration
- Local storage: `storage/` directory
- S3 integration: Optional, configured via environment variables
- File cleanup: Automatic cleanup of temporary files
- **S3 Configuration**: Requires Signature Version 4 (automatically configured)

## 🧪 Testing

Use the included `sample_dataset.csv` for testing:

**Fast Mode (Default):**
```bash
curl -X POST "http://localhost:8000/dataset/upload" \
  -F "file=@sample_dataset.csv" \
  -F "user_id=test_user" \
  -F "mode=fast"
```

**Smart Mode:**
```bash
curl -X POST "http://localhost:8000/dataset/upload" \
  -F "file=@sample_dataset.csv" \
  -F "user_id=test_user" \
  -F "mode=smart"
```

**Deep Mode:**
```bash
curl -X POST "http://localhost:8000/dataset/upload" \
  -F "file=@sample_dataset.csv" \
  -F "user_id=test_user" \
  -F "mode=deep"
```

**Get Cleaning Report (Deep Mode Only):**
```bash
curl "http://localhost:8000/dataset/{dataset_id}/report"
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License

```
MIT License

Copyright (c) 2024 NeuraLearn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🆘 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the processing logs via API endpoints
3. Check database logs for detailed error information

---

**Version**: 1.0.0  
**API Documentation**: Available at `/docs` when running