from gann import *
import tflowtools as TFT
import time

class GannMan:
    ganns = None

    def __init__(self):
        self.ganns = []

    def create_gann(self, name, networkDimsString, hiddenActivationFunc, outputActivationFunc,
                    lossFunc, optimizer, learningRate, weightInit, dataSource, dSourcePars, caseFrac, valFrac,
                    testFrac, miniBatchSize):

        # Convert strings of numbers to list of ints
        if weightInit != "scaled": weightInit = [float(i) for i in weightInit.split(" ")]
        networkDims = [int(i) for i in networkDimsString.split(" ")]
        if dSourcePars == '': dSourcePars = []
        else: dSourcePars = [int(i) for i in dSourcePars.split(" ")]

        # Generate cases for the data source
        case_generator = None
        if(dataSource == 'bitcounter'):
            case_generator = (lambda: TFT.gen_vector_count_cases(dSourcePars[0], dSourcePars[1]))
        elif(dataSource == 'autoencoder'):
            case_generator = (lambda: TFT.gen_all_one_hot_cases(dSourcePars[0]))
        elif(dataSource == 'dense_autoencoder'):
            case_generator = (lambda: TFT.gen_dense_autoencoder_cases(dSourcePars[0], dSourcePars[1], dr=(0.4, 0.7)))
        elif(dataSource == 'parity'):
            case_generator = (lambda: TFT.gen_all_parity_cases(dSourcePars[0]))
        elif(dataSource == 'segment'):
            case_generator = (lambda: TFT.gen_segmented_vector_cases(size=25, count=1000, misegs=0, maxsegs=8))
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
                   optimizer = optimizer, weightRange = weightInit)
        #ann.run(epochs = 200, showInterval = 200, validationInterval = 10, bestk = 1)
        self.ganns.append(ann)


    def run_network(self, network, showInterval = 100, validationInterval = 100, epochs=100, sess=None,
                    mapBatchSize = '', displayWeights = '', displayBiases = '', continued=False,
                    mapDendrograms = '', bestk=None):
        if mapBatchSize == '': mapBatchSize = 0
        else: mapBatchSize = int(mapBatchSize)
        if mapDendrograms == '': mapDendrograms = []
        else: mapDendrograms = [int(i) for i in mapDendrograms.split(" ")]
        if displayWeights == '': displayWeights = []
        else: displayWeights = [int(i) for i in displayWeights.split(" ")]
        if displayBiases == '': displayBiases = []
        else: displayBiases = [int(i) for i in displayBiases.split(" ")]
        network.run(epochs=epochs, showInterval=showInterval, validationInterval=validationInterval, bestk=bestk,
                    )#displayBiases = displayBiases, displayWeights = displayWeights,
                    #mapDendrograms = mapDendrograms, mapBatchSize = mapBatchSize)
        exit = input("Press anything to continue ..")

    def run_network_more(self, epochs, name):
        #TODO: do gann.runmore()
        pass

    def load_best_param_networks(self, fileName):
        dataSource = fileName
        name = fileName + '_best'
        with open('best_param_networks/' + fileName + '.txt', 'r') as f:
            for paramLine in f:
                paramLine = paramLine.strip("\n")
                paramLine = paramLine.split(",")
                paramName = paramLine[0]
                paramLine.pop(0)
                if paramName == 'netDims': netDims = paramLine[0]
                elif paramName == 'hiddenActivFunc': hiddenActivationFunc = paramLine[0]
                elif paramName == 'outputActivFunc': outputActivationfunc = paramLine[0]
                elif paramName == 'lossFunc': lossFunc = paramLine[0]
                elif paramName == 'optimizer': optimizer = paramLine[0]
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
                    lossFunc, optimizer, learningRate, weightInit, fileName, dSourceParams, caseFrac, valFrac,
                    testFrac, mbs)

#man = GannMan()
#man.load_best_param_networks(fileName='bitcounter')