from gann import *
import tflowtools as TFT
import time

class GannMan:
    ganns = None

    def __init__(self):
        self.ganns = []

    def create_gann(self, name, networkDimsString, hiddenActivationFunc, outputActivationFunc,
                    lossFunc, optimizer, learningRate, weightInit, dataSource, dataSourceParas, caseFrac, valFrac,
                    testFrac, miniBatchSize, mapBatchSize, steps, mapLayers, mapDendograms,
                    displayWeights, displayBiases):

        # Convert strings of numbers to list of ints
        if weightInit != "scaled": weightInit = [float(i) for i in weightInit.split(" ")]
        networkDims = [int(i) for i in networkDimsString.split(" ")]
        if dataSourceParas == '': dataSourceParas = []
        else: dataSourceParas = [int(i) for i in dataSourceParas.split(" ")]
        if displayWeights == '': displayWeights = []
        else: displayWeights = [int(i) for i in displayWeights.split(" ")]
        if displayBiases == '': displayBiases = []
        else: displayBiases = [int(i) for i in displayBiases.split(" ")]

        # Generate cases and layers for specific data source cases
        if(dataSource == 'bitcounter'):
            case_generator = (lambda: TFT.gen_vector_count_cases(dataSourceParas[0], dataSourceParas[1]))
        else:
            raise NotImplementedError("Datasource: " + dataSource + " is not implemented")

        cMan = Caseman(cfunc = case_generator, cfrac=float(caseFrac),
                       vfrac=float(valFrac), tfrac=float(testFrac))
        ann = Gann(name = name, netDims = networkDims, cMan = cMan, learningRate = float(learningRate),
                   mbs = int(miniBatchSize), hiddenActivationFunc = hiddenActivationFunc,
                   outputActivationFunc = outputActivationFunc, lossFunc = lossFunc,
                   optimizer = optimizer, weightRange = weightInit, displayBiases = displayBiases,
                   displayWeights = displayWeights)
        #ann.run(epochs = 200, showInterval = 200, validationInterval = 10, bestk = 1)
        self.ganns.append(ann)


    def run_network(self, network, showInterval = 100, validationInterval = 100, epochs=100, sess=None, continued=False, bestk=None):
        network.run(epochs=epochs, showInterval=showInterval, validationInterval=validationInterval, bestk=bestk)
        exit = input("Press anything to continue ..")

    def run_network_more(self, epochs, name):
        #TODO: do gann.runmore()
        pass

    def run_best_param_case(self):
        pass