from gann import *
import tflowtools as TFT
import time

class GannMan:
    gann = None

    def __init__(self):
        self.gann = None

    def __del__(self):
        pass

    def create_gann(self, name, networkDimsString, hiddenActivationFunc, outputActivationFunc,
                    lossFunc, optimizer, momentumFrac, learningRate, weightInit, dataSource, dSourcePars, caseFrac, valFrac,
                    testFrac, miniBatchSize):

        # Convert strings of numbers to list of ints
        if weightInit != "scaled": weightInit = [float(i) for i in weightInit.split(" ")]
        networkDims = [int(i) for i in networkDimsString.split(" ")]
        if dSourcePars == '': dSourcePars = []
        else: dSourcePars = [int(i) for i in dSourcePars.split(" ")]
        if momentumFrac == None: pass
        else: momentumFrac = float(momentumFrac)

        # Generate cases for the data source
        case_generator = None
        if(dataSource == 'bitcounter'):
            case_generator = (lambda: TFT.gen_vector_count_cases(dSourcePars[0], dSourcePars[1]))
        elif(dataSource == 'autoencoder'):
            case_generator = (lambda: TFT.gen_all_one_hot_cases(dSourcePars[0]))
        elif(dataSource == 'dense_autoencoder'):
            case_generator = (lambda: TFT.gen_dense_autoencoder_cases(dSourcePars[0], dSourcePars[1], dr=(0.1, 0.9)))
        elif(dataSource == 'parity'):
            case_generator = (lambda: TFT.gen_all_parity_cases(dSourcePars[0]))
        elif(dataSource == 'segment'):
            case_generator = (lambda: TFT.gen_segmented_vector_cases(vectorlen=25, count=1000, minsegs=0, maxsegs=8))
        elif(dataSource == 'MNIST'):
            pass
        elif(dataSource == 'wine'):
            pass
        elif(dataSource == 'glass'):
            pass
        elif(dataSource == 'yeast'):
            pass
        elif(dataSource == 'hackers'):
            pass
        else:
            raise NotImplementedError("Datasource: " + dataSource + " is not implemented")

        cMan = Caseman(cfunc = case_generator, cfrac=float(caseFrac),
                       vfrac=float(valFrac), tfrac=float(testFrac))
        ann = Gann(name = name, netDims = networkDims, cMan = cMan, learningRate = float(learningRate),
                   mbs = int(miniBatchSize), hiddenActivationFunc = hiddenActivationFunc,
                   outputActivationFunc = outputActivationFunc, lossFunc = lossFunc,
                   optimizer = optimizer, momentum = momentumFrac, weightRange = weightInit)
        #ann.run(epochs = 200, showInterval = 200, validationInterval = 10, bestk = 1)
        self.gann = ann


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
        self.gann.run(epochs=epochs, showInterval=showInterval, validationInterval=validationInterval, bestk=bestK,
                    displayWeights = displayWeights, displayBiases = displayBiases)#displayBiases = displayBiases, displayWeights = displayWeights,
                    #mapDendrograms = mapDendrograms, mapBatchSize = mapBatchSize)
        self.gann.run_mapping(mapBatchSize = mapBatchSize, mapDendrograms = mapDendrograms, mapLayers = mapLayers)

    def do_gann_from_config(self, fileName):
        dataSource = fileName
        name = fileName + '_best'
        momentumFrac = None
        with open('best_param_networks/' + fileName + '.txt', 'r') as f:
            for paramLine in f:
                paramLine = paramLine.strip("\n")
                paramLine = paramLine.split(",")
                if(paramLine[0] == ''): continue
                elif(paramLine[0][0] == '#'): continue #Skip comments and empty lines
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
                    elif paramName == 'momentumFrac': momentumFrac = paramLine[0]
                    elif paramName == 'learningRate': learningRate = paramLine[0]
                    elif paramName == 'weightInit': weightInit = paramLine[0]
                    elif paramName == 'dSourceParams': dSourceParams = paramLine[0]
                    elif paramName == 'caseFrac': caseFrac = paramLine[0]
                    elif paramName == 'valFrac': valFrac = paramLine[0]
                    elif paramName == 'testFrac': testFrac = paramLine[0]
                    elif paramName == 'mbs': mbs = paramLine[0]
                    else:
                        raise AssertionError("Parameter: " + paramName + ", is not a valid parameter name.")
        self.create_gann(name, netDims, hiddenActivationFunc, outputActivationfunc,
                    lossFunc, optimizer, momentumFrac, learningRate, weightInit, dataSource, dSourceParams, caseFrac, valFrac,
                    testFrac, mbs)

        self.run_gann(epochs=int(epochs), showInterval=None,
                                 validationInterval=int(valInt), bestK=bestK,
                                 mapBatchSize=mapBatchSize,
                                 mapLayers=mapLayers, mapDendrograms=mapDendrograms,
                                 displayWeights=displayWeights, displayBiases=displayBiases)

#man = GannMan()
#man.do_gann_from_config(fileName='dense_autoencoder')

