# NeuraLearn Project Structure

## Root Directory Layout
```
neuralearn-microservices/
├── main.py                 # FastAPI application entry point
├── database.py            # SQLAlchemy models and database operations
├── s3_client.py           # AWS S3 integration utilities
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment configuration
├── .env                  # Environment variables (not in git)
├── .gitignore           # Git ignore patterns
├── services/            # Business logic layer
├── storage/             # Local file storage
└── docs/                # Documentation files
```

## Services Directory (`services/`)
**Purpose**: Contains all business logic separated from API layer following service layer pattern.

```
services/
├── __init__.py
├── data_loader.py        # Dataset loading from files/URLs
├── cleaner.py           # AI-powered data cleaning
├── eda.py               # Exploratory data analysis
├── outliers.py          # Outlier detection and handling
├── feature_engineering.py # Feature transformation and encoding
└── gemini_client.py     # Google Gemini API integration
```

### Service Responsibilities
- **DataLoader**: File I/O, format detection, dataset persistence
- **DataCleaner**: Missing value handling with AI suggestions
- **EDAService**: Visualization generation and statistical analysis
- **OutlierService**: Outlier detection (IQR/Z-score) and correction
- **FeatureEngineeringService**: Categorical encoding, scaling, feature selection
- **GeminiClient**: AI API integration with structured prompts

## Storage Directory (`storage/`)
**Purpose**: Local file system storage with organized subdirectories.

```
storage/
├── datasets/            # Raw and cleaned CSV files
│   ├── {dataset_id}_raw.csv
│   └── {dataset_id}_cleaned.csv
└── graphs/              # EDA visualizations
    ├── {dataset_id}_histogram_{column}.png
    ├── {dataset_id}_boxplot_{column}.png
    ├── {dataset_id}_correlation_heatmap.png
    ├── {dataset_id}_countplot_{column}.png
    └── {dataset_id}_summary_stats.json
```

### File Naming Conventions
- **Dataset Files**: `{uuid}_raw.csv`, `{uuid}_cleaned.csv`
- **Visualizations**: `{uuid}_{graph_type}_{column}.png`
- **Statistics**: `{uuid}_summary_stats.json`
- **UUIDs**: Used for unique identification and collision prevention

## Database Schema Organization

### Core Entities
- **Dataset**: Main entity with metadata and processing status
- **Processing Logs**: Separate tables for cleaning, outlier, and feature operations
- **Graph Metadata**: References to generated visualizations

### Relationship Patterns
- One-to-many relationships with cascade deletes
- Foreign key constraints for data integrity
- Indexed columns for query performance (userId, collaborationId)

## Configuration Management

### Environment Variables (`.env`)
```
DATABASE_URL=postgresql://...     # Required: PostgreSQL connection
GEMINI_API_KEY=...               # Optional: AI features
AWS_ACCESS_KEY_ID=...            # Optional: S3 storage
AWS_SECRET_ACCESS_KEY=...        # Optional: S3 storage
AWS_REGION=...                   # Optional: S3 region
S3_BUCKET_NAME=...               # Optional: S3 bucket
```

### Configuration Hierarchy
1. Environment variables (highest priority)
2. `.env` file
3. Default values in code (lowest priority)

## API Layer Organization

### Endpoint Grouping
- **Dataset Management**: Upload, list, download
- **Processing Logs**: Cleaning, outliers, features
- **Visualizations**: EDA graphs and metadata
- **Collaboration**: Multi-user dataset sharing

### Response Models
- Pydantic models for type safety and validation
- Consistent error response format
- Background task integration for long-running operations

## Processing Pipeline Architecture

### Async Processing Flow
1. **Synchronous**: Upload validation and dataset creation
2. **Background Task**: Data processing pipeline
3. **Status Updates**: Database status tracking
4. **Result Storage**: Local and cloud storage

### Error Handling Strategy
- Graceful degradation (S3 → local storage)
- AI fallback (Gemini API → heuristics)
- Comprehensive error logging
- Status tracking for user feedback

## Code Organization Patterns

### Dependency Injection
- Database sessions injected via FastAPI dependency system
- Service instances created per request
- Configuration loaded from environment

### Separation of Concerns
- **API Layer** (`main.py`): HTTP handling, validation, routing
- **Service Layer** (`services/`): Business logic, data processing
- **Data Layer** (`database.py`): Persistence, queries, transactions
- **Integration Layer** (`s3_client.py`, `gemini_client.py`): External services

### Error Boundaries
- Service-level exception handling
- API-level error response formatting
- Background task error isolation
- Logging at appropriate levels

## File Lifecycle Management

### Temporary Files
- Created in system temp directory during upload
- Automatically cleaned up after processing
- Error handling for cleanup failures

### Persistent Storage
- Raw datasets preserved for audit trail
- Cleaned datasets for download
- Visualizations for repeated access
- Optional S3 migration for scalability

## Development Conventions

### Code Style
- Python type hints throughout
- Async/await for I/O operations
- Descriptive variable and function names
- Comprehensive docstrings for public APIs

### Testing Structure (Future)
```
tests/
├── unit/
│   ├── test_services/
│   ├── test_database.py
│   └── test_api.py
├── integration/
│   ├── test_pipeline.py
│   └── test_s3_integration.py
└── fixtures/
    ├── sample_datasets/
    └── mock_responses/
```

### Logging Strategy
- Structured logging with context
- Different log levels for different environments
- Processing step tracking
- Performance metrics collection

This structure promotes maintainability, scalability, and clear separation of responsibilities while supporting both local development and production deployment scenarios.