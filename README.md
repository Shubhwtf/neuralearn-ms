# NeuraLearn Microservices API

A comprehensive FastAPI-based microservice for automated data processing, cleaning, exploratory data analysis (EDA), and feature engineering. This service provides intelligent data preprocessing capabilities with AI-powered suggestions using Google's Gemini API.

## 🚀 Features

- **Automated Data Processing Pipeline**: Upload datasets and get them automatically cleaned, analyzed, and feature-engineered
- **Redis-Based Job Queue**: Scalable background job processing with Redis and RQ for horizontal scaling
- **Automatic Worker Scaling**: Built-in autoscaler that adjusts worker count based on queue length
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
- **Real-time Monitoring**: Queue monitoring tools for tracking job processing and system health

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
│   ├── queue.py           # Redis queue management
│   └── schema.py          # Pydantic models for API responses
├── jobs/                  # Background job workers
│   └── eda_worker.py      # EDA processing worker function
├── scripts/               # Utility scripts
│   ├── autoscaler.py      # Automatic worker scaling
│   ├── monitor_redis.py   # Queue monitoring
│   ├── load_test.py       # Load testing tool
│   └── check_redis.sh     # Redis health check
└── storage/               # Local file storage
    ├── datasets/          # Raw and cleaned datasets
    └── graphs/            # Generated EDA visualizations
```

### Queue Architecture

```
API Server (FastAPI)
    ↓
Redis Queue (RQ)
    ├── eda_fast (fast/smart mode jobs)
    └── eda_deep (deep mode jobs)
        ↓
Workers (RQ Workers)
    ├── Worker 1 (eda_fast)
    ├── Worker 2 (eda_fast)
    └── Worker N (auto-scaled)
        ↓
Processing Pipeline
    ├── Data Cleaning
    ├── EDA Generation
    ├── Outlier Detection
    └── Feature Engineering
```

## 📋 Requirements

- Python 3.8+
- PostgreSQL database
- Redis server (for job queue)
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

# Redis (for job queue)
REDIS_URL=redis://localhost:6379/0

# Google Gemini API (optional)
GEMINI_API_KEY=your_gemini_api_key

# AWS S3 (optional)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
S3_BUCKET_NAME=your_bucket_name
```

5. **Start Redis**
```bash
# Using Docker (recommended)
docker run --name redis -p 6379:6379 -d redis:7-alpine

# Or using system Redis
sudo systemctl start redis-server  # Ubuntu/Debian
brew services start redis          # macOS
```

6. **Start the API server**
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

7. **Start workers** (choose one option)

**Option A: Manual Workers**
```bash
# Terminal 1: Fast queue worker
rq worker eda_fast

# Terminal 2: Deep queue worker (optional)
rq worker eda_deep
```

**Option B: Autoscaler (Recommended)**
```bash
# Automatically manages workers based on queue length
python scripts/autoscaler.py
```

The autoscaler will:
- Maintain minimum workers (1 for `eda_fast`, 0 for `eda_deep`)
- Scale up when queue > 50 jobs
- Scale down when queue < 10 jobs
- Manage worker processes automatically

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
4. **Job Enqueueing**: Processing job added to Redis queue (`eda_fast` or `eda_deep`)
5. **Background Processing** (handled by workers):
   - **Cleaning**: Mode-specific missing value handling
     - **Fast**: Heuristic-based cleaning (mean, median, mode, forward-fill)
     - **Smart**: Limited AI assistance for columns with >20% nulls (up to 3 Gemini calls)
     - **Deep**: Full AI reasoning for top 3 columns with >10% nulls (1 Gemini call with detailed report)
   - **EDA**: Automatic generation of visualizations and statistics
   - **Outlier Detection**: IQR/Z-score based outlier detection and fixing
   - **Feature Engineering**: Categorical encoding, scaling, derived features
   - **Report Generation**: Deep mode generates comprehensive cleaning report
6. **Completion**: Cleaned dataset and metadata available for download

### Queue Processing

- Jobs are automatically routed to appropriate queues:
  - `eda_fast`: Fast and smart mode jobs (10-minute timeout)
  - `eda_deep`: Deep mode jobs (30-minute timeout)
- Workers process jobs asynchronously
- Status updates are tracked in the database
- Failed jobs are retried automatically (up to 3 times with exponential backoff)

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

### Quick Start Script

Use the provided startup script:
```bash
./scripts/start_all.sh
```

This will:
- Start Redis if not running
- Provide instructions for starting API and workers

### Heroku Deployment
The project includes a `Procfile` for Heroku deployment:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Note**: For Heroku, you'll need:
- Redis addon (e.g., Heroku Redis)
- Worker dynos for processing jobs
- Set `REDIS_URL` environment variable

### Docker Deployment

**Docker Compose Example:**
```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
  
  worker:
    build: .
    command: rq worker eda_fast eda_deep
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    deploy:
      replicas: 3  # Scale workers
```

**Dockerfile:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

- **Redis**: Use managed Redis (AWS ElastiCache, Redis Cloud, etc.)
- **Workers**: Use process managers (systemd, supervisor) or orchestration (Kubernetes, Docker Swarm)
- **Monitoring**: Set up alerts for queue length and worker health
- **Scaling**: Use autoscaler or orchestration tools for automatic scaling

## 🔧 Configuration

### Data Processing Limits
- Maximum dataset size: 100,000 rows × 100 columns
- Outlier percentage threshold: 10%
- **Cleaning Mode Limits:**
  - Fast mode: Unlimited
  - Smart mode: 3 calls per user
  - Deep mode: 1 call per user

### Queue Configuration
- **Queue Names**: `eda_fast` (fast/smart), `eda_deep` (deep)
- **Job Timeouts**: 
  - Fast/Smart: 10 minutes
  - Deep: 30 minutes
- **Retry Policy**: Up to 3 retries with exponential backoff (1m, 2m, 5m)
- **Result TTL**: 0 (results discarded immediately to save memory)
- **Failure TTL**: 24 hours (failed jobs kept for debugging)

### Storage Configuration
- Local storage: `storage/` directory
- S3 integration: Optional, configured via environment variables
- File cleanup: Automatic cleanup of temporary files
- **S3 Configuration**: Requires Signature Version 4 (automatically configured)

### Worker Configuration
- **Minimum Workers**: 1 for `eda_fast`, 0 for `eda_deep`
- **Maximum Workers**: Configurable (default: 20 for fast, 5 for deep)
- **Autoscaling**: Enabled by default with autoscaler script
- **Worker Isolation**: Each worker runs in separate process

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

### Queue Monitoring

**Real-time Queue Monitoring:**
```bash
# Continuous monitoring (updates every 5 seconds)
python scripts/monitor_redis.py --watch

# One-time check
python scripts/monitor_redis.py
```

**Quick Redis Check:**
```bash
./scripts/check_redis.sh
```

**Check Job Status:**
```bash
# Check database status of datasets
python scripts/check_job_status.py
```

### Monitoring Features

- **Queue Statistics**: Jobs queued, started, finished, failed per queue
- **Worker Information**: Active workers, their queues, current jobs
- **Redis Server Metrics**: Memory usage, connection count, cache hit rate
- **Processing Status**: Track dataset status via API endpoints
- **Comprehensive Logging**: All operations logged to database

### Load Testing

Test the queue system at scale:
```bash
# Basic load test (50 datasets, 10 concurrent)
python scripts/load_test.py

# Large scale test
python scripts/load_test.py --num 200 --concurrent 30 --mode fast
```

### Autoscaler Configuration

Customize autoscaling behavior:
```bash
# More aggressive scaling
python scripts/autoscaler.py --scale-up 30 --scale-down 5 --workers-per-scale 2

# Conservative scaling
python scripts/autoscaler.py --scale-up 100 --scale-down 20 --interval 15
```

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

## 🎯 Queue System Benefits

### Scalability
- **Horizontal Scaling**: Add more workers to increase throughput
- **Load Distribution**: Jobs distributed across multiple workers
- **Queue Isolation**: Fast jobs don't block deep jobs (separate queues)

### Reliability
- **Job Persistence**: Jobs survive worker restarts (stored in Redis)
- **Automatic Retries**: Failed jobs retry with exponential backoff
- **Failure Tracking**: Failed jobs logged for debugging

### Performance
- **Non-blocking API**: API responds immediately after enqueueing
- **Parallel Processing**: Multiple workers process jobs concurrently
- **Resource Management**: Workers can be scaled based on load

### Monitoring
- **Real-time Visibility**: Monitor queue lengths and worker status
- **Performance Metrics**: Track processing times and throughput
- **Health Checks**: Monitor Redis and worker health

## 🆘 Support & Troubleshooting

### Common Issues

**Jobs not processing:**
- Check Redis is running: `redis-cli ping`
- Verify workers are running: `python scripts/monitor_redis.py`
- Check worker logs for errors

**Queue growing too fast:**
- Add more workers: `rq worker eda_fast` (multiple terminals)
- Or use autoscaler: `python scripts/autoscaler.py`
- Check job processing time (may need optimization)

**Redis connection errors:**
- Verify Redis is running: `docker ps | grep redis`
- Check `REDIS_URL` in `.env` matches Redis location
- Test connection: `redis-cli -u $REDIS_URL ping`

**Workers not scaling:**
- Check autoscaler is running
- Verify queue thresholds are appropriate
- Check worker logs for startup errors

### Getting Help

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the processing logs via API endpoints
3. Check database logs for detailed error information
4. Monitor queues: `python scripts/monitor_redis.py --watch`
5. Check job status: `python scripts/check_job_status.py`

---

**Version**: 1.0.0  
**API Documentation**: Available at `/docs` when running