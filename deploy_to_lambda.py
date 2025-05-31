#!/usr/bin/env python3
"""
Deployment helper script for Lambda Labs GPU Cloud
This script shows you the commands to run for deploying your training job to Lambda Labs.

Before running this script, make sure you have:
1. Created a Lambda Labs account at https://lambda.ai/
2. Added your SSH key to your Lambda Labs account
3. Launched a GPU instance (recommended: 1x NVIDIA A6000 or higher)
4. Note down the instance IP address
"""

import os
import sys

def print_instructions():
    print("=" * 60)
    print("LAMBDA LABS DEPLOYMENT INSTRUCTIONS")
    print("=" * 60)
    
    print("\n1. First, get your Lambda Labs instance IP address from the dashboard")
    print("   - Go to https://cloud.lambdalabs.com/instances")
    print("   - Launch a GPU instance (recommended: 1x NVIDIA A6000 or A100)")
    print("   - Note the IP address shown")
    
    print("\n2. Transfer your project files to the instance:")
    instance_ip = input("\nEnter your Lambda Labs instance IP address: ").strip()
    
    if not instance_ip:
        print("Error: Please provide the instance IP address")
        return
    
    ssh_key_path = input("Enter path to your SSH key (e.g., ~/.ssh/lambda_key.pem): ").strip()
    if not ssh_key_path:
        ssh_key_path = "~/.ssh/lambda_key.pem"
    
    print(f"\n3. Run these commands in your terminal:\n")
    
    print("# Create project directory on remote instance")
    print(f'ssh -i {ssh_key_path} ubuntu@{instance_ip} "mkdir -p ~/neural_seq_decoder"')
    
    print("\n# Transfer project files")
    print(f'scp -i {ssh_key_path} -r src/ ubuntu@{instance_ip}:~/neural_seq_decoder/')
    print(f'scp -i {ssh_key_path} -r scripts/ ubuntu@{instance_ip}:~/neural_seq_decoder/')
    print(f'scp -i {ssh_key_path} -r speechBCI_data/ ubuntu@{instance_ip}:~/neural_seq_decoder/')
    print(f'scp -i {ssh_key_path} requirements.txt ubuntu@{instance_ip}:~/neural_seq_decoder/')
    
    print("\n# Connect to the instance")
    print(f'ssh -i {ssh_key_path} ubuntu@{instance_ip}')
    
    print("\n4. Once connected to the instance, run these commands:")
    print("""
cd ~/neural_seq_decoder

# Install dependencies
pip install -r requirements.txt

# Create logs directory
mkdir -p logs/speech_logs

# Run the training (this will take some time!)
python scripts/train_model_cloud.py

# Optional: Run in background with output logging
nohup python scripts/train_model_cloud.py > training.log 2>&1 &

# Optional: Monitor progress
tail -f training.log
""")
    
    print("\n5. Important reminders:")
    print("   - Remember to TERMINATE your instance when training is complete!")
    print("   - You'll be charged by the hour until you terminate the instance")
    print("   - Download your trained models before terminating:")
    print(f'   scp -i {ssh_key_path} -r ubuntu@{instance_ip}:~/neural_seq_decoder/logs/ .')
    
    print("\n6. Monitor your training:")
    print("   - Lambda instances come with Jupyter pre-installed")
    print("   - You can access it at http://[INSTANCE_IP]:8888")
    print("   - Default password is usually provided in the instance details")
    
    print(f"\n7. Estimated costs (as of 2024):")
    print("   - 1x NVIDIA A6000: ~$0.80/hour")
    print("   - 1x NVIDIA A100: ~$1.29/hour") 
    print("   - 1x NVIDIA H100: ~$2.49-3.29/hour")
    print("   - Your training should take a few hours depending on nBatch setting")

if __name__ == "__main__":
    print_instructions() 