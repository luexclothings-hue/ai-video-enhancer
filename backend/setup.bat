@echo off
REM Setup script for Windows local development

echo Setting up AI Video Enhancer Backend...

REM Check Node.js
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Node.js not found. Please install Node.js 20+
    exit /b 1
)
echo Node.js found

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.10+
    exit /b 1
)
echo Python found

REM Setup API
echo.
echo Setting up API...
cd apps\api

if not exist .env (
    echo Creating .env file...
    copy .env.example .env
    echo Please edit apps\api\.env with your configuration
)

echo Installing dependencies...
call npm install

echo Generating Prisma client...
call npm run generate

echo API setup complete
cd ..\..

REM Setup Worker
echo.
echo Setting up Worker...
cd apps\worker

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

if not exist .env (
    echo Creating .env file...
    copy .env.example .env
    echo Please edit apps\worker\.env with your configuration
)

echo Worker setup complete
cd ..\..

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Edit apps\api\.env with your database and GCP credentials
echo 2. Edit apps\worker\.env with your configuration
echo 3. Run database migrations: cd apps\api ^&^& npm run migrate:dev
echo 4. Clone Stream-DiffVSR: cd apps\worker ^&^& git clone https://github.com/jamichss/Stream-DiffVSR.git stream_diffvsr
echo 5. Start API: cd apps\api ^&^& npm run dev
echo 6. Start Worker: cd apps\worker ^&^& .venv\Scripts\activate.bat ^&^& python main.py
echo.
echo Documentation: http://localhost:3000/documentation

pause
