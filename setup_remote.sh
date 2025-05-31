#!/bin/bash

# Setup script for Lambda Labs GPU instance
# Run this script on the remote Lambda Labs instance

echo "Setting up neural_seq_decoder on Lambda Labs GPU instance..."

# Update system packages
echo "Updating system packages..."
sudo apt update

# Install Python pip if not available
echo "Installing pip..."
sudo apt install -y python3-pip

# Install project dependencies
echo "Installing Python dependencies..."
cd ~/neural_seq_decoder
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p logs/speech_logs

# Check CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')"

echo "Setup complete! You can now run:"
echo "python scripts/train_model_cloud.py"
echo ""
echo "To run in background with logging:"
echo "nohup python scripts/train_model_cloud.py > training.log 2>&1 &"
echo ""
echo "To monitor progress:"
echo "tail -f training.log" 