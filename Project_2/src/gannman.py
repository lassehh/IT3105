#import gann
import tflowtools as TFT

class GannMan:
    ganns = None

    def __init__(self):
        self.ganns = []

    def create_gann(self, name, networkDims, hiddenActivationFunc, outputActivationFunc,
                    lossFunc, learningRate, weightInit, dataSource, dataSourceParas, caseFrac, valFrac,
                    miniBatchSize, mapBatchSize, steps, mapLayers, mapDendograms,
                    displayWeights, displayBiases):
        #TODO: create case_gen and a gann similator to autoex and countex in gann.py
        pass

    def run_network(self, epochs, name):
        #TODO: train, validate, test and show graphs etc for a given network (gann.run())
        pass

    def run_network_more(self, epochs, name):
        #TODO: do gann.runmore()
        pass

    def run_best_param_case(self):
        pass