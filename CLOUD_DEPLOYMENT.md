# Training Neural Seq Decoder on Lambda Labs GPU Cloud

This guide will help you train your neural sequence decoder model on Lambda Labs GPU cloud infrastructure.

## Prerequisites

1. **Lambda Labs Account**: Create an account at [https://lambda.ai/](https://lambda.ai/)
2. **SSH Key**: Set up SSH key authentication with Lambda Labs
3. **Dataset**: Ensure your `speechBCI_data/pickled_data.pkl` file is ready

## Quick Start

### Step 1: Prepare for Deployment
Run the deployment helper script locally:
```bash
python deploy_to_lambda.py
```
This will guide you through the deployment process.

### Step 2: Launch Lambda Labs Instance
1. Go to [Lambda Cloud Console](https://cloud.lambdalabs.com/instances)
2. Click "Launch Instance"
3. **Recommended configurations**:
   - **For budget training**: 1x NVIDIA A6000 (~$0.80/hour)
   - **For faster training**: 1x NVIDIA A100 (~$1.29/hour)
   - **For fastest training**: 1x NVIDIA H100 (~$2.49/hour)
4. Select your SSH key
5. Launch the instance and note the IP address

### Step 3: Transfer Files
Replace `<INSTANCE_IP>` with your instance IP and `<SSH_KEY_PATH>` with your SSH key path:

```bash
# Create project directory
ssh -i <SSH_KEY_PATH> ubuntu@<INSTANCE_IP> "mkdir -p ~/neural_seq_decoder"

# Transfer files
scp -i <SSH_KEY_PATH> -r src/ ubuntu@<INSTANCE_IP>:~/neural_seq_decoder/
scp -i <SSH_KEY_PATH> -r scripts/ ubuntu@<INSTANCE_IP>:~/neural_seq_decoder/
scp -i <SSH_KEY_PATH> -r speechBCI_data/ ubuntu@<INSTANCE_IP>:~/neural_seq_decoder/
scp -i <SSH_KEY_PATH> requirements.txt ubuntu@<INSTANCE_IP>:~/neural_seq_decoder/
scp -i <SSH_KEY_PATH> setup_remote.sh ubuntu@<INSTANCE_IP>:~/neural_seq_decoder/
```

### Step 4: Connect and Setup
```bash
# Connect to instance
ssh -i <SSH_KEY_PATH> ubuntu@<INSTANCE_IP>

# Run setup script
cd ~/neural_seq_decoder
bash setup_remote.sh
```

### Step 5: Start Training
```bash
# Option 1: Run interactively (blocks terminal)
python scripts/train_model_cloud.py

# Option 2: Run in background with logging (recommended)
nohup python scripts/train_model_cloud.py > training.log 2>&1 &

# Monitor progress (for Option 2)
tail -f training.log
```

### Step 6: Monitor Training
- Training progress will be printed every 100 batches
- Check GPU utilization: `nvidia-smi`
- Monitor log file: `tail -f training.log`

### Step 7: Download Results and Cleanup

**📥 Download your trained models** (see [DOWNLOADING_MODELS.md](DOWNLOADING_MODELS.md) for detailed guide):

```bash
# Download trained models (run from your LOCAL machine)
scp -i <SSH_KEY_PATH> -r ubuntu@<INSTANCE_IP>:~/neural_seq_decoder/logs/ .
```

**🚨 IMPORTANT: Terminate instance to stop billing!**
- Go to Lambda Cloud Console and terminate the instance

## What Gets Saved During Training

Your training will create these important files:

### 🎯 **Model Weights** (Most Important)
- `modelWeights_best` - Best performing model (lowest CER)
- `modelWeights_final` - Final model from last batch  
- `modelWeights` - Copy of best weights (for compatibility)

### 📊 **Training Data**
- `args` - Training configuration/hyperparameters
- `trainingStats` - Loss and CER history for analysis

### ❌ **You DON'T Need to Download**
- `speechBCI_data/` - You already have this dataset locally
- Any temporary files or logs (unless you want them)

## Training Configuration

The cloud training script (`scripts/train_model_cloud.py`) uses these settings:
- **Model**: speechBaseline4
- **Batch size**: 64
- **Training batches**: 10,000
- **GPU memory**: Optimized for single GPU training
- **Output**: Saved to `./logs/speech_logs/speechBaseline4/`

## Cost Estimation

Based on 10,000 training batches (~3-5 hours):
- **A6000**: ~$2.40-4.00
- **A100**: ~$3.87-6.45  
- **H100**: ~$7.47-16.45

## Troubleshooting

### CUDA Not Available
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Out of Memory Errors
Reduce batch size in `train_model_cloud.py`:
```python
args['batchSize'] = 32  # or 16
```

### Connection Issues
- Ensure your SSH key is properly configured
- Check that the instance is running in the Lambda console
- Verify the IP address is correct

### Slow Training
- Use a higher-tier GPU (A100 or H100)
- Monitor GPU utilization with `nvidia-smi`
- Check if data loading is the bottleneck

## Important Reminders

⚠️ **ALWAYS TERMINATE YOUR INSTANCE** when done to stop billing!

⚠️ **DOWNLOAD YOUR MODELS** before terminating the instance!

⚠️ **MONITOR COSTS** in the Lambda Labs dashboard.

## Loading Your Trained Model Locally

Once downloaded, load your model:

```python
import sys
sys.path.append('src')
from neural_decoder.neural_decoder_trainer_cloud import loadModel

# Load the best model
model = loadModel('./logs/speech_logs/speechBaseline4/', device='cuda')
```

## Support

- Lambda Labs Documentation: [https://docs.lambdalabs.com/](https://docs.lambdalabs.com/)
- Lambda Labs Support: Available through their console
- Community Forum: [https://community.lambdalabs.com/](https://community.lambdalabs.com/)

## Next Steps

1. **📖 Read**: [DOWNLOADING_MODELS.md](DOWNLOADING_MODELS.md) for detailed download instructions
2. **🔍 Analyze**: Review training curves and model performance
3. **🚀 Deploy**: Use your trained model for inference
4. **🔄 Iterate**: Experiment with different hyperparameters 