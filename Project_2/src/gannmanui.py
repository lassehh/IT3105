import os
from gannman import *
import time
from msvcrt import getwch

class GannManUi:
    state = None
    gannMan = None
    pointer = '>>'
    menuIndexPointer = 0
    stateDict = {
        0: 'readScenario',
        1: 'selectSavedScenario',
        2: 'loadBest',
        3: 'exit'
    }
    headerText = '--- GANN UI FOR PROJECT 2 DEMO ---'
    initMenuOptions = {
        0: '1. Input scenario parameters to build a network.',
        1: '2. Select a saved scenario to run/runmore/do mappings.',
        2: '3. Select a datasource and load a scenario with the best known parameters',
        3: '4. Exit program.'}

    def __init__(self, state = 'init'):
        self.state = state
        self.gannMan = GannMan()

    def init_menu(self):
        optionSelected = False
        menuLength = len(self.initMenuOptions)
        while(not optionSelected):
            os.system('cls' if os.name == 'nt' else 'clear')
            print(self.headerText)
            print("--- MAIN MENU ---")
            for i in range(0,menuLength):
                if i == self.menuIndexPointer:
                    print(self.pointer + '\t' + self.initMenuOptions[i])
                else:
                    print('\t' + self.initMenuOptions[i])
            character = getwch()
            if character == "w": self.menuIndexPointer = (self.menuIndexPointer - 1) % menuLength
            elif character == "s": self.menuIndexPointer = (self.menuIndexPointer + 1) % menuLength
            elif character == '\r':
                optionSelected = True
                self.state = self.stateDict[self.menuIndexPointer]
            time.sleep(0.01)



    def read_scenario_menu(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.headerText)
        print("--- INPUT SCENARIO ---")
        print("Specify scenario parameters ...")
        name = input("Network name (for later reference): ")
        networkDims = input("Network dimensions [l1 l2 l3]: ")
        hiddenActivationFunc = "relu"#input("Hidden activation function (relu, sigmoid or tanh): ")
        outputActivationFunc = "softmax"#input("Output activation function (softmax, none): ")
        lossFunc = "softmax_cross_entropy"#input("Loss function (MSE, softmax_cross_entropy, sigmoid_cross_entropy): ")
        optimizer = "momentum"#input("Optimizer (gradient_descent or momentum): ")
        learningRate = input("Learning rate <0, 1>: ")
        weightInit = "-0.1 0.1"#input("Initial weight range (or scaled): ")
        dataSource = "bitcounter"#input("Data source (bitcounter, autoencoder, MNIST...): ")
        dataSourceParas = "500 15"#input("Data source parameters (Ex: nbits for bit-counter): ")
        caseFrac = "1"#input("Case fraction: ")
        valFrac = "0.1"#input("Validation fraction: ")
        testFrac = "0.1"#input("Final testing fraction: ")
        miniBatchSize = input("Mini batch size: ")

        self.gannMan.create_gann(name, networkDims, hiddenActivationFunc, outputActivationFunc,
                                 lossFunc, optimizer, learningRate, weightInit, dataSource, dataSourceParas,
                                 caseFrac, valFrac, testFrac, miniBatchSize)

        waitForExit = input("\nPRESS ENTER TO GO BACK MENU..")
        self.state = "init"

    def select_created_scenario(self):
        # Print header and selectable scenarios to screen
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.headerText)
        print("--- CREATED SCENARIOS--- ")
        print("\nNUMBER \t\t NAME")
        for i, network in enumerate(self.gannMan.ganns):
            print(str(i) + ' \t\t ' + network.name)
        print('')
        # User-input for action
        selectName = input("Choose network: ")
        selectAction = input("Action (run, runmore): ")
        selectEpochs = input("Epochs: ")
        selectShowInt = input("Show interval: ")
        selectValInt = input("Validation interval: ")
        selectBestK = input("Best K: ")
        selectMapBatchSize = input("Map batch size: ")
        selectSteps = input("Steps/Number of mbs to be run through the system in training: ")
        selectMapLayers = input("Map layers (indices): ")
        selectMapDendograms = input("Map dendograms (indices): ")
        selectDisplayWeights = input("Weights to be visualized: ")
        selectDisplayBiases = input("Biases to be visualized: ")
        # Run the network: training and testing
        if selectBestK == '': selectBestK = None
        for _, network in enumerate(self.gannMan.ganns):
            if network.name == selectName:
                if selectAction == "run":
                    self.gannMan.run_network(network = network, epochs = int(selectEpochs), showInterval = int(selectShowInt),
                            validationInterval = int(selectValInt), bestk = int(selectBestK))
                elif selectAction == "runmore":
                    network.runmore(epochs = int(selectEpochs), bestk = int(selectBestK))
            break
        waitForExit = input("\nPRESS ENTER TO GO BACK MENU..")
        self.state = "init"

    def load_best_param_scenario(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.headerText)
        print("--- BEST PARAMETER SCENARIOS--- \n")
        print("NAME")
        print("bitcounter")

        selectDataSource = input("\nSelect data source: ")
        self.gannMan.load_best_param_networks(selectDataSource)
        waitForExit = input("\nPRESS ENTER TO GO BACK MENU..")
        self.state = "init"

    def run(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        exit = False
        while(not exit):
            if(self.state == 'init'): self.init_menu()
            if(self.state == 'readScenario'): self.read_scenario_menu()
            if(self.state == 'selectSavedScenario'): self.select_created_scenario()
            if (self.state == 'loadBest'): self.load_best_param_scenario()
            if(self.state == "exit"):
                exit = True
                print("\nExiting program ..")
                time.sleep(1.2)





if __name__ == '__main__':
    ui = GannManUi()
    #ui.read_scenario_menu()
    #ui.select_created_scenario()
    #ui.load_best_param_scenario()
    ui.run()# doesn't work with debugging, instead run the function you want to debuf directly
    #ex: read_scenario_menu() to experiment with different networks parameters