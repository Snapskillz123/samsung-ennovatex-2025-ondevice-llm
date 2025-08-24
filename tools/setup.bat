@echo off
REM Samsung EnnovateX 2025 - Windows Setup Script

echo 🚀 Setting up Samsung EnnovateX 2025 project...

REM Check Python version
echo 📋 Checking Python version...
python --version
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Create virtual environment
echo 🔧 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing requirements...
pip install -r requirements.txt

REM Create data directories
echo 📁 Creating data directories...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\communication 2>nul
mkdir data\calendar 2>nul
mkdir data\notes 2>nul
mkdir out\models 2>nul
mkdir out\adapters 2>nul
mkdir models\base 2>nul
mkdir models\adapters 2>nul

REM Create sample WhatsApp export
echo 📄 Creating sample data files...
echo 8/20/2025, 10:30 AM - Alice: Hey, how's your day going? > data\raw\sample_whatsapp.txt
echo 8/20/2025, 10:32 AM - Bob: Pretty good! Just working on that project. You? >> data\raw\sample_whatsapp.txt
echo 8/20/2025, 10:33 AM - Alice: Same here! Coffee later? ☕ >> data\raw\sample_whatsapp.txt
echo 8/20/2025, 10:35 AM - Bob: Sounds great! The usual place at 3? >> data\raw\sample_whatsapp.txt
echo 8/20/2025, 10:36 AM - Alice: Perfect! See you there 👍 >> data\raw\sample_whatsapp.txt
echo 8/20/2025, 2:45 PM - Bob: Running 5 mins late, traffic is crazy >> data\raw\sample_whatsapp.txt
echo 8/20/2025, 2:46 PM - Alice: No worries, I'll grab us a table >> data\raw\sample_whatsapp.txt

echo 🧪 Testing installation...
python -c "
import sys
from pathlib import Path
src_path = Path('.') / 'src'
sys.path.insert(0, str(src_path))

try:
    from data_pipeline.parse_whatsapp import process_whatsapp_file
    from adapters.protocol import AdapterManager
    from adapters.router import create_router
    print('✅ All core modules imported successfully')
    
    router = create_router()
    result = router.route('Hey, how are you?', ui_context='messaging')
    print(f'✅ Router test passed: {result.adapter_ids}')
    print('🎉 Setup complete!')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo.
echo 🎉 Setup complete!
echo.
echo 📋 Next steps:
echo 1. Add your WhatsApp chat exports to data/raw/
echo 2. Process data: python src/data_pipeline/parse_whatsapp.py --input data/raw/chat.txt --output data/processed/chat.jsonl
echo 3. Apply privacy filter: python src/data_pipeline/privacy_filter.py --input data/processed/chat.jsonl --output data/processed/filtered.jsonl
echo 4. Train adapter: python src/training/train_qlora.py --config configs/communication_adapter.json --train-data data/processed/filtered.jsonl
echo.
echo 📖 See README.md for detailed instructions
pause
