import time
import tflowtools as TFT
from gann import *

""" 
Class: GannMan
Handles all the user-specified parameters from the GannManUi to create and run scenarios.

Dependencies: gann, tflowtools

"""
class GannMan:
    ann = None                 # The general neural network which a GannMan object manages

    def __init__(self):
        self.ann = None

    def __del__(self):
        pass

    # Description: Formats the parameters passed to this function and then creates a gann with the specified
    #           parameters.
    # Input: All parameters needed for a nn architecture.
    # Output: None
    def create_gann(self, name, networkDimsString, hiddenActivationFunc, outputActivationFunc,
                    lossFunc, optimizer, optimizerParams,learningRate, weightInitType, weightInit, dataSource, dSourcePars, caseFrac, valFrac,
                    testFrac, miniBatchSize):

        # Convert strings of numbers to list of ints
        if weightInitType == "uniform": weightInit = [float(i) for i in weightInit.split(" ")]
        else: weightInit = (0,0)
        networkDims = [int(i) for i in networkDimsString.split(" ")]
        if dSourcePars == '': dSourcePars = []
        else: dSourcePars = [int(i) for i in dSourcePars.split(" ")]
        if optimizerParams == '': optimizerParams = []
        else: optimizerParams = [float(i) for i in optimizerParams.split(" ")]

        # Find the correct case-generating function
        case_generator = None
        if(dataSource == 'bitcounter'):
            case_generator = (lambda: TFT.gen_vector_count_cases(dSourcePars[0], dSourcePars[1]))
        elif(dataSource == 'autoencoder'):
            case_generator = (lambda: TFT.gen_all_one_hot_cases(dSourcePars[0]))
        elif(dataSource == 'dense_autoencoder'):
            case_generator = (lambda: TFT.gen_dense_autoencoder_cases(count = dSourcePars[0], size = dSourcePars[1],
                                                                      dr=(dSourcePars[2], dSourcePars[3])))
        elif(dataSource == 'parity'):
            case_generator = (lambda: TFT.gen_all_parity_cases(dSourcePars[0]))
        elif(dataSource == 'segment'):
            case_generator = (lambda: TFT.gen_segmented_vector_cases(vectorlen = dSourcePars[0], count = dSourcePars[1], minsegs = dSourcePars[2], maxsegs = dSourcePars[3]))
        elif(dataSource == 'MNIST'):
            case_generator = (lambda: TFT.gen_mnist_cases())
        elif(dataSource == 'wine'):
            case_generator = (lambda: TFT.gen_uc_irvine_cases('winequality_red'))
        elif(dataSource == 'glass'):
            case_generator = (lambda: TFT.gen_uc_irvine_cases('glass'))
        elif(dataSource == 'yeast'):
            case_generator = (lambda: TFT.gen_uc_irvine_cases('yeast'))
        elif(dataSource == 'hackers'):
            case_generator = (lambda: TFT.gen_hackers_choice_cases('balance-scale'))
        else:
            raise NotImplementedError("Datasource: " + dataSource + " is not implemented")

        cMan = Caseman(cfunc = case_generator, cfrac=float(caseFrac),
                       vfrac=float(valFrac), tfrac=float(testFrac))
        ann = Gann(name = name, netDims = networkDims, cMan = cMan, learningRate = float(learningRate),
                   mbs = int(miniBatchSize), hiddenActivationFunc = hiddenActivationFunc,
                   outputActivationFunc = outputActivationFunc, lossFunc = lossFunc,
                   optimizer = optimizer, optimizerParams = optimizerParams, weightInitType = weightInitType, weightRange = weightInit)
        self.ann = ann

    # Description: Formats the parameters passed to this function and then runs the gann with the specified
    #           parameters.
    # Input: All parameters needed to run a gann
    # Output: None
    def run_gann(self, showInterval = None, validationInterval = 100, epochs=100, sess=None,
                    mapBatchSize = '', displayWeights = '', displayBiases = '', continued=False,
                    mapLayers = '', mapDendrograms = '', bestK=None):
        if mapBatchSize == '': mapBatchSize = 0
        else: mapBatchSize = int(mapBatchSize)
        if mapDendrograms == '': mapDendrograms = []
        else: mapDendrograms = [int(i) for i in mapDendrograms.split(" ")]
        if mapLayers == '': mapLayers = []
        else: mapLayers = [int(i) for i in mapLayers.split(" ")]
        if displayWeights == '': displayWeights = []
        else: displayWeights = [int(i) for i in displayWeights.split(" ")]
        if displayBiases == '': displayBiases = []
        else: displayBiases = [int(i) for i in displayBiases.split(" ")]
        if bestK == 'none': bestK = None
        else: bestK = int(bestK)
        self.ann.run(epochs=epochs, showInterval=showInterval, validationInterval=validationInterval, bestk=bestK,
                     displayWeights = displayWeights, displayBiases = displayBiases)
        self.ann.run_mapping(mapBatchSize = mapBatchSize, mapDendrograms = mapDendrograms, mapLayers = mapLayers)

    # Description: Reads all the neccessary parameters from a predefined file for a network scenario and passes
    #           them to create_gann and run_Ggann.
    # Input: filename
    # Output: None
    def do_gann_from_config(self, fileName):
        name = fileName
        with open('best_param_networks/' + fileName, 'r') as f:
            for paramLine in f:
                paramLine = paramLine.strip("\n")
                paramLine = paramLine.split(",")
                if(paramLine[0] == ''): continue # Skip empty lines
                elif(paramLine[0][0] == '#'): continue #Skip comments
                else:
                    paramName = paramLine[0]
                    paramLine.pop(0)
                    # Run parameters
                    if paramName == 'epochs': epochs = paramLine[0]
                    elif paramName == 'valInt': valInt = paramLine[0]
                    elif paramName == 'bestK': bestK = paramLine[0]
                    elif paramName == 'mapBatchSize': mapBatchSize = paramLine[0]
                    elif paramName == 'mapLayers': mapLayers = paramLine[0]
                    elif paramName == 'mapDendrograms': mapDendrograms= paramLine[0]
                    elif paramName == 'displayWeights': displayWeights = paramLine[0]
                    elif paramName == 'displayBiases': displayBiases = paramLine[0]
                    # Creation parameters
                    elif paramName == 'netDims': netDims = paramLine[0]
                    elif paramName == 'hiddenActivFunc': hiddenActivationFunc = paramLine[0]
                    elif paramName == 'outputActivFunc': outputActivationfunc = paramLine[0]
                    elif paramName == 'lossFunc': lossFunc = paramLine[0]
                    elif paramName == 'optimizer': optimizer = paramLine[0]
                    elif paramName == 'optimizerParams': optimizerParams = paramLine[0]
                    elif paramName == 'learningRate': learningRate = paramLine[0]
                    elif paramName == 'weightInitType': weightInitType = paramLine[0]
                    elif paramName == 'weightInit': weightInit = paramLine[0]
                    elif paramName == 'dataSource': dataSource = paramLine[0]
                    elif paramName == 'dSourceParams': dSourceParams = paramLine[0]
                    elif paramName == 'caseFrac': caseFrac = paramLine[0]
                    elif paramName == 'valFrac': valFrac = paramLine[0]
                    elif paramName == 'testFrac': testFrac = paramLine[0]
                    elif paramName == 'mbs': mbs = paramLine[0]
                    else:
                        raise AssertionError("Parameter: " + paramName + ", is not a valid parameter name.")
        self.create_gann(name, netDims, hiddenActivationFunc, outputActivationfunc,
                    lossFunc, optimizer, optimizerParams, learningRate, weightInitType, weightInit, dataSource, dSourceParams, caseFrac, valFrac,
                    testFrac, mbs)

        self.run_gann(epochs=int(epochs), showInterval=None,
                                 validationInterval=int(valInt), bestK=bestK,
                                 mapBatchSize=mapBatchSize,
                                 mapLayers=mapLayers, mapDendrograms=mapDendrograms,
                                 displayWeights=displayWeights, displayBiases=displayBiases)
