modelName = 'speechBaseline4'

args = {}
# Use relative paths for cloud deployment
args['outputDir'] = './logs/speech_logs/' + modelName
args['datasetPath'] = './speechBCI_data/pickled_data.pkl'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

# Add the src directory to Python path
import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_decoder.neural_decoder_trainer import trainModel

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(args['outputDir'], exist_ok=True)
    
    print("=" * 60)
    print(f"STARTING TRAINING: {modelName}")
    print("=" * 60)
    print(f"Output directory: {args['outputDir']}")
    print(f"Dataset path: {args['datasetPath']}")
    print(f"Batch size: {args['batchSize']}")
    print(f"Total batches: {args['nBatch']}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # Run training
    trainModel(args)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print("Saved files in output directory:")
    
    output_files = []
    if os.path.exists(args['outputDir']):
        for file in os.listdir(args['outputDir']):
            file_path = os.path.join(args['outputDir'], file)
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            output_files.append((file, file_size_mb))
            print(f"  - {file} ({file_size_mb:.2f} MB)")
    
    print("\nTo download your trained model, run this from your LOCAL machine:")
    print(f"scp -i <SSH_KEY_PATH> -r ubuntu@<INSTANCE_IP>:~/neural_seq_decoder/logs/ .")
    print("\nIMPORTANT: Remember to TERMINATE your Lambda Labs instance to stop billing!")
    print("=" * 60) 