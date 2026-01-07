@echo off
REM AI Video Enhancer - Development Setup Script (Windows)
REM This script sets up the development environment

echo 🚀 Setting up AI Video Enhancer Development Environment
echo ==================================================

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "temp" mkdir temp

REM Check if GCP service account key exists
if not exist "gcp-key.json" (
    echo ⚠️  GCP service account key not found!
    echo Please download your GCP service account key and save it as 'gcp-key.json'
    echo You can create one at: https://console.cloud.google.com/iam-admin/serviceaccounts
    echo.
    echo For development, creating a dummy file to continue:
    echo {"type": "service_account"} > gcp-key.json
    echo ✅ Created dummy GCP key file for development
)

REM Set up API environment
echo 🔧 Setting up API environment...
cd apps\api

if not exist ".env" (
    copy .env.example .env >nul
    echo ✅ Created API .env file from example
    echo ⚠️  Please edit apps\api\.env with your actual values
)

REM Install API dependencies
echo 📦 Installing API dependencies...
call npm install

REM Generate Prisma client
echo 🔄 Generating Prisma client...
call npx prisma generate

cd ..\..

REM Set up Worker environment
echo 🔧 Setting up Worker environment...
cd apps\worker

if not exist ".env" (
    copy .env.example .env >nul
    echo ✅ Created Worker .env file from example
    echo ⚠️  Please edit apps\worker\.env with your actual values
)

cd ..\..

REM Check for Stream-DiffVSR setup
echo 🤖 Stream-DiffVSR Model Setup...
echo ✅ Stream-DiffVSR will automatically download from HuggingFace Hub on first use
echo 📦 Model size: ~2GB (cached after first download)
echo 🌐 Ensure internet connection is available for first run

REM Build and start services
echo 🐳 Building Docker containers...
docker-compose build

echo 🚀 Starting database...
docker-compose up -d postgres

REM Wait for database to be ready
echo ⏳ Waiting for database to be ready...
timeout /t 10 /nobreak >nul

echo.
echo ✅ Development environment setup complete!
echo.
echo 🎯 Next steps:
echo 1. Edit apps\api\.env with your GCP credentials
echo 2. Edit apps\worker\.env with your GCP credentials
echo 3. Start all services: docker-compose up
echo 4. Stream-DiffVSR will auto-download on first use (~2GB)
echo.
echo 📚 Useful commands:
echo   docker-compose up          # Start all services
echo   docker-compose logs -f     # View logs
echo   docker-compose down        # Stop all services
echo.
echo 🌐 Services will be available at:
echo   API: http://localhost:3000
echo   API Docs: http://localhost:3000/documentation
echo   Database: localhost:5432

pause