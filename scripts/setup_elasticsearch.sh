#!/bin/bash
# Setup Elasticsearch locally via Docker for the BM25 pipeline.
# Usage: ./scripts/setup_elasticsearch.sh

set -e

echo "🔍 Setting up Elasticsearch 8.x for BM25 RAG pipeline..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Run Elasticsearch
docker run -d \
    --name vectorless-rag-es \
    -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
    elasticsearch:8.13.0

echo "⏳ Waiting for Elasticsearch to start..."
sleep 15

# Health check
if curl -s http://localhost:9200/_cluster/health | grep -q '"status"'; then
    echo "✅ Elasticsearch is running at http://localhost:9200"
else
    echo "❌ Elasticsearch failed to start. Check: docker logs vectorless-rag-es"
    exit 1
fi
