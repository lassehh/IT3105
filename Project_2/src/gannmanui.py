import os
from gannman import *
import time
from msvcrt import getwch

class GannManUi:
    state = None
    gannMan = None
    pointer = '>>'
    menuIndexPointer = 0
    headerText = '--- GANN UI FOR PROJECT 2 DEMO ---\n'
    stateDict = {0: 'inputScenario', 1: 'loadScenario', 2: 'runScenario', 3: 'exit'}
    initMenuOptions = { 0: 'INPUT scenario', 1: 'LOAD scenario', 2: 'RUN scenario', 3: 'Exit program.'}

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
        hiddenActivationFunc = input("Hidden activation function (relu, sigmoid or tanh): ")
        outputActivationFunc = input("Output activation function (softmax, none): ")
        lossFunc = input("Loss function (MSE, softmax_cross_entropy, sigmoid_cross_entropy): ")
        optimizer = input("Optimizer (gradient_descent or momentum): ")
        if optimizer == "momentum":
            momentumFrac= input("Amount of momentum <0,1>: ")
        else:
            momentumFrac = None
        learningRate = input("Learning rate <0, 1>: ")
        weightInit = input("Initial weight range (or scaled): ")
        dataSource = input("Data source (bitcounter, autoencoder, MNIST...): ")
        dataSourceParas = input("Data source parameters (Ex: nbits for bit-counter): ")
        caseFrac = input("Case fraction: ")
        valFrac = input("Validation fraction: ")
        testFrac = input("Final testing fraction: ")
        miniBatchSize = input("Mini batch size: ")

        self.gannMan.create_gann(name, networkDims, hiddenActivationFunc, outputActivationFunc,
                                 lossFunc, optimizer, momentumFrac, learningRate, weightInit, dataSource, dataSourceParas,
                                 caseFrac, valFrac, testFrac, miniBatchSize)

        waitForExit = input("\nPRESS ENTER TO GO BACK MENU..")
        self.state = "init"

    def run_scenario_menu(self):
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
        #selectSteps = input("Steps/Number of mbs to be run through the system in training: ")
        selectMapLayers = input("Map layers (indices): ")
        selectMapDendrograms = input("Map dendograms (indices): ")
        selectDisplayWeights = input("Weights to be visualized: ")
        selectDisplayBiases = input("Biases to be visualized: ")
        # Run the network: training and testing
        if selectBestK == '': selectBestK = None
        for _, network in enumerate(self.gannMan.ganns):
            if network.name == selectName:
                if selectAction == "run":
                    self.gannMan.run_network(network = network, epochs = int(selectEpochs), showInterval = int(selectShowInt),
                            validationInterval = int(selectValInt), bestk = int(selectBestK), mapBatchSize = selectMapBatchSize,
                            mapLayers = selectMapLayers, mapDendrograms = selectMapDendrograms,
                            displayWeights = selectDisplayWeights, displayBiases = selectDisplayBiases)
                elif selectAction == "runmore":
                    network.runmore(epochs = int(selectEpochs), bestk = int(selectBestK))
            break
        waitForExit = input("\nPRESS ENTER TO GO BACK MENU..")
        self.state = "init"

    def load_scenario_menu(self):
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
            if(self.state == 'inputScenario'): self.read_scenario_menu()
            if(self.state == 'runScenario'): self.run_scenario_menu()
            if (self.state == 'loadScenario'): self.load_scenario_menu()
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