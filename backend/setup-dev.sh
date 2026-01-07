#!/bin/bash

# AI Video Enhancer - Development Setup Script
# This script sets up the development environment

set -e

echo "🚀 Setting up AI Video Enhancer Development Environment"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p temp

# Check if GCP service account key exists
if [ ! -f "gcp-key.json" ]; then
    echo "⚠️  GCP service account key not found!"
    echo "Please download your GCP service account key and save it as 'gcp-key.json'"
    echo "You can create one at: https://console.cloud.google.com/iam-admin/serviceaccounts"
    echo ""
    echo "For development, you can create a dummy file to continue:"
    echo '{"type": "service_account"}' > gcp-key.json
    echo "✅ Created dummy GCP key file for development"
fi

# Set up API environment
echo "🔧 Setting up API environment..."
cd apps/api

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created API .env file from example"
    echo "⚠️  Please edit apps/api/.env with your actual values"
fi

# Install API dependencies
echo "📦 Installing API dependencies..."
npm install

# Generate Prisma client
echo "🔄 Generating Prisma client..."
npx prisma generate

cd ../..

# Set up Worker environment
echo "🔧 Setting up Worker environment..."
cd apps/worker

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created Worker .env file from example"
    echo "⚠️  Please edit apps/worker/.env with your actual values"
fi

cd ../..

# Check for Stream-DiffVSR setup
echo "🤖 Stream-DiffVSR Model Setup..."
echo "✅ Stream-DiffVSR will automatically download from HuggingFace Hub on first use"
echo "📦 Model size: ~2GB (cached after first download)"
echo "🌐 Ensure internet connection is available for first run"

# Build and start services
echo "🐳 Building Docker containers..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d postgres

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Run database migrations
echo "🗄️  Running database migrations..."
docker-compose exec api npx prisma migrate deploy

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Edit apps/api/.env with your GCP credentials"
echo "2. Edit apps/worker/.env with your GCP credentials"
echo "3. Start all services: docker-compose up"
echo "4. Stream-DiffVSR will auto-download on first use (~2GB)"
echo ""
echo "📚 Useful commands:"
echo "  docker-compose up          # Start all services"
echo "  docker-compose logs -f     # View logs"
echo "  docker-compose down        # Stop all services"
echo "  docker-compose exec api sh # Access API container"
echo ""
echo "🌐 Services will be available at:"
echo "  API: http://localhost:3000"
echo "  API Docs: http://localhost:3000/documentation"
echo "  Database: localhost:5432"