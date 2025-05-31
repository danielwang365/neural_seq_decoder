
modelName = 'speechBaseline4'

args = {}
args['outputDir'] = 'C:/Users/danie/OneDrive/Documents/GitHub/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = 'C:/Users/danie/OneDrive/Documents/GitHub/neural_seq_decoder/speechBCI_data/pickled_data.pkl'
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

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)