#!/bin/bash
# Start all services for Redis queue system

set -e

echo "🚀 Starting NeuraLearn Queue System"
echo "===================================="
echo ""

# Check if Redis is running
if ! docker ps | grep -q redis; then
    echo "📦 Starting Redis..."
    docker run --name redis -p 6379:6379 -d redis:7-alpine 2>/dev/null || docker start redis
    sleep 2
    echo "✅ Redis started"
else
    echo "✅ Redis already running"
fi

echo ""
echo "💡 Next steps:"
echo ""
echo "1. Start API server:"
echo "   uvicorn main:app --reload"
echo ""
echo "2. Choose ONE option:"
echo ""
echo "   Option A - Manual workers:"
echo "   rq worker eda_fast"
echo "   rq worker eda_deep"
echo ""
echo "   Option B - Autoscaler (recommended):"
echo "   python scripts/autoscaler.py"
echo ""
echo "3. Monitor queues:"
echo "   python scripts/monitor_redis.py --watch"
echo ""
