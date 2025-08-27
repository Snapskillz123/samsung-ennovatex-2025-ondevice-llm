#!/bin/bash

# Samsung EnnovateX 2025 - Quick Setup Script
# Sets up the complete development environment

set -e  # Exit on any error

echo "🚀 Setting up Samsung EnnovateX 2025 project..."

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python --version 2>&1)
echo "Found: $python_version"

if ! python -c "import sys; assert sys.version_info >= (3, 9)" 2>/dev/null; then
    echo "❌ Error: Python 3.9+ is required"
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{raw,processed,communication,calendar,notes}
mkdir -p out/{models,adapters}
mkdir -p models/{base,adapters}

# Create sample data structure
echo "📄 Creating sample data files..."

# Create sample WhatsApp export format
cat > data/raw/sample_whatsapp.txt << 'EOF'
8/20/2025, 10:30 AM - Alice: Hey, how's your day going?
8/20/2025, 10:32 AM - Bob: Pretty good! Just working on that project. You?
8/20/2025, 10:33 AM - Alice: Same here! Coffee later? ☕
8/20/2025, 10:35 AM - Bob: Sounds great! The usual place at 3?
8/20/2025, 10:36 AM - Alice: Perfect! See you there 👍
8/20/2025, 2:45 PM - Bob: Running 5 mins late, traffic is crazy
8/20/2025, 2:46 PM - Alice: No worries, I'll grab us a table
EOF

# Create development script
cat > quick_start.py << 'EOF'
"""Quick start script for development and testing."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    print("🔧 Samsung EnnovateX 2025 - Development Environment")
    print("=" * 50)
    
    try:
        # Test imports
        from data_pipeline.parse_whatsapp import main as parse_main
        from adapters.protocol import AdapterManager, AdapterMeta
        from adapters.router import create_router
        print("✅ All core modules imported successfully")
        
        # Test data pipeline
        print("📊 Testing data pipeline...")
        from data_pipeline.parse_whatsapp import process_whatsapp_file
        
        input_file = Path("data/raw/sample_whatsapp.txt")
        output_file = Path("data/processed/sample_parsed.jsonl")
        
        if input_file.exists():
            num_pairs = process_whatsapp_file(input_file, output_file, max_pairs=10)
            print(f"✅ Processed {num_pairs} conversation pairs")
        else:
            print("⚠️  Sample data not found, create some WhatsApp exports in data/raw/")
        
        # Test router
        print("🧭 Testing adapter router...")
        router = create_router()
        result = router.route("Hey, how are you doing?", ui_context="messaging")
        print(f"✅ Router selected adapters: {result.adapter_ids}")
        
        print("\n🎉 Development environment is ready!")
        print("\nNext steps:")
        print("1. Add your WhatsApp exports to data/raw/")
        print("2. Run: python src/data_pipeline/parse_whatsapp.py --input data/raw/your_export.txt --output data/processed/parsed.jsonl")
        print("3. Run: python src/training/train_qlora.py --config configs/communication_adapter.json --train-data data/processed/parsed.jsonl")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
EOF

echo "🧪 Testing installation..."
python quick_start.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add your WhatsApp chat exports to data/raw/"
echo "2. Process data: python src/data_pipeline/parse_whatsapp.py --input data/raw/chat.txt --output data/processed/chat.jsonl"
echo "3. Apply privacy filter: python src/data_pipeline/privacy_filter.py --input data/processed/chat.jsonl --output data/processed/filtered.jsonl"
echo "4. Train adapter: python src/training/train_qlora.py --config configs/communication_adapter.json --train-data data/processed/filtered.jsonl"
echo ""
echo "💡 Run 'python quick_start.py' anytime to test your setup"
echo "📖 See README.md for detailed instructions"
