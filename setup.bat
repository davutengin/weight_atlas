@echo off
echo WeightAtlas Setup
echo =================

echo.
echo [1/2] Installing Python dependencies...
cd backend
pip install -r requirements.txt
cd ..

echo.
echo [2/2] Building frontend...
cd frontend
npm install
npm run build
cd ..

echo.
echo Setup complete! Run: python weightatlas.py
pause
