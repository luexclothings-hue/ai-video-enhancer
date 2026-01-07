#!/bin/bash
# Setup script for local development

set -e

echo "🚀 Setting up AI Video Enhancer Backend..."

# Check prerequisites
echo "Checking prerequisites..."

# Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 20+"
    exit 1
fi
echo "✅ Node.js $(node --version)"

# Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi
echo "✅ Python $(python3 --version)"

# PostgreSQL
if ! command -v psql &> /dev/null; then
    echo "⚠️  PostgreSQL client not found. Make sure PostgreSQL is installed."
fi

# FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found. Worker will need FFmpeg."
fi

# Setup API
echo ""
echo "📦 Setting up API..."
cd apps/api

if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit apps/api/.env with your configuration"
fi

echo "Installing dependencies..."
npm install

echo "Generating Prisma client..."
npm run generate

echo "✅ API setup complete"

# Setup Worker
echo ""
echo "🤖 Setting up Worker..."
cd ../worker

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit apps/worker/.env with your configuration"
fi

echo "✅ Worker setup complete"

cd ../..

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit apps/api/.env with your database and GCP credentials"
echo "2. Edit apps/worker/.env with your configuration"
echo "3. Run database migrations: cd apps/api && npm run migrate:dev"
echo "4. Clone Stream-DiffVSR: cd apps/worker && git clone https://github.com/jamichss/Stream-DiffVSR.git stream_diffvsr"
echo "5. Start API: cd apps/api && npm run dev"
echo "6. Start Worker: cd apps/worker && source .venv/bin/activate && python main.py"
echo ""
echo "📚 Documentation: http://localhost:3000/documentation"
