# Testing & Monitoring Scripts

Scripts for load testing and monitoring the Redis queue system.

## Prerequisites

Install additional dependencies:
```bash
pip install aiohttp
```

## Load Testing

### Basic Usage

Run a load test with default settings (50 datasets, 10 concurrent):
```bash
python scripts/load_test.py
```

### Advanced Options

```bash
# Upload 100 datasets with 20 concurrent uploads
python scripts/load_test.py --num 100 --concurrent 20

# Test deep mode
python scripts/load_test.py --mode deep --num 10

# Custom user ID
python scripts/load_test.py --user my_test_user --num 30
```

### Options

- `--num`: Number of datasets to upload (default: 50)
- `--concurrent`: Number of concurrent uploads (default: 10)
- `--mode`: Processing mode - `fast`, `smart`, or `deep` (default: `fast`)
- `--user`: User ID for testing (default: `load_test_user`)

### What It Tests

1. **Upload Phase**: Tests API's ability to handle concurrent uploads
2. **Queue Phase**: Verifies jobs are properly enqueued to Redis
3. **Processing Phase**: Monitors job completion rates

### Expected Output

```
🚀 Starting load test:
   - Datasets to upload: 50
   - Concurrent uploads: 10
   - Mode: fast
   - API: http://localhost:8000

📤 Uploading datasets...
✅ Upload phase complete in 12.34s
   - Successful uploads: 50/50

📊 Monitoring job processing...
   [5s] Status: uploaded=10, processing=35, completed=5, failed=0
   [10s] Status: uploaded=0, processing=20, completed=30, failed=0
   ...

📈 LOAD TEST SUMMARY
   Upload Phase:
     ✅ Successful: 50/50
     ⏱️  Total time: 12.34s
     📊 Throughput: 4.05 uploads/sec
```

## Monitoring

### One-Time Check

```bash
python scripts/monitor_redis.py
```

### Continuous Monitoring (Watch Mode)

```bash
# Refresh every 5 seconds (default)
python scripts/monitor_redis.py --watch

# Custom refresh interval
python scripts/monitor_redis.py --watch --interval 10
```

### What It Shows

- **Queue Statistics**: Jobs queued, started, finished, failed per queue
- **Worker Information**: Active workers, their queues, current jobs
- **Redis Server Info**: Memory usage, connection count, cache hit rate

### Expected Output

```
======================================================================
📊 REDIS QUEUE MONITORING - 2025-01-27 10:30:45
======================================================================

📋 QUEUE STATISTICS:
----------------------------------------------------------------------

  Queue: eda_fast
    📤 Queued:        25
    ⚙️  Started:       10
    ✅ Finished:     150
    ❌ Failed:         2
    👷 Workers:        3
    ⏰ Oldest job:   45.2s ago
    📊 Total jobs:   187

  Queue: eda_deep
    📤 Queued:         2
    ⚙️  Started:        1
    ✅ Finished:       5
    ❌ Failed:         0
    👷 Workers:        1
    📊 Total jobs:      8

👷 WORKER INFORMATION:
----------------------------------------------------------------------

  Worker: rq:worker:abc123
    Queues:     eda_fast
    State:      busy
    Current job: job:abc123:dataset-456
    Started:    2025-01-27T10:00:00

🔧 REDIS SERVER:
----------------------------------------------------------------------
  Connected clients:  5
  Memory used:        2.5M
  Commands processed: 12500
  Cache hit rate:     95.2%
```

## Testing Workflow

### 1. Start Infrastructure

```bash
# Terminal 1: Redis
docker run --name redis -p 6379:6379 redis:7-alpine

# Terminal 2: Fast workers (2-3 workers)
rq worker eda_fast
rq worker eda_fast  # In another terminal

# Terminal 3: Deep worker
rq worker eda_deep

# Terminal 4: API server
uvicorn main:app --reload
```

### 2. Run Load Test

```bash
# Terminal 5: Load test
python scripts/load_test.py --num 100 --concurrent 20
```

### 3. Monitor in Real-Time

```bash
# Terminal 6: Monitoring
python scripts/monitor_redis.py --watch --interval 3
```

## Interpreting Results

### Good Signs ✅

- **High throughput**: >5 uploads/sec
- **Low queue backlog**: Queued jobs < 50 for extended periods
- **Fast processing**: Jobs complete within expected timeouts
- **Low failure rate**: <5% failed jobs
- **Worker utilization**: Workers stay busy but not overloaded

### Warning Signs ⚠️

- **Growing queue**: Queued jobs continuously increasing
- **High failure rate**: >10% failed jobs
- **Slow processing**: Jobs taking longer than expected
- **Worker starvation**: Workers idle while jobs are queued

### Scale Indicators 📈

- **Horizontal scaling needed**: Queue backlog > 100 consistently
- **More workers needed**: Workers always busy, queue growing
- **Redis bottleneck**: High memory usage or connection limits
- **Database bottleneck**: Slow status updates despite fast processing

## Troubleshooting

### Redis Connection Error

```
❌ Error: Could not connect to Redis!
```

**Solution**: Check `REDIS_URL` in `.env` and ensure Redis is running.

### No Workers Found

```
⚠️  No active workers found!
```

**Solution**: Start workers with `rq worker eda_fast` and `rq worker eda_deep`.

### Jobs Not Processing

**Check**:
1. Workers are running and connected to Redis
2. Queue names match (`eda_fast` vs `eda_deep`)
3. Workers are listening to correct queues
4. No errors in worker logs

### High Memory Usage

**Solution**: 
- Clear finished jobs: `rq info --url redis://localhost:6379/0`
- Increase Redis memory limit
- Use `result_ttl=0` (already configured) to avoid storing results
