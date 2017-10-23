import os
from gannman import *
import time
from msvcrt import getwch

class GannManUi:
    state = None
    gannMan = None
    pointer = '>>'
    menuIndexPointer = 0
    headerText = '< GANN MANAGER UI - PROJECT 2 DEMO >\n'
    stateDict = {0: 'inputRunScenario', 1: 'loadRunScenario', 2: 'exit'}
    initMenuOptions = { 0: 'INPUT & RUN scenario', 1: 'LOAD & RUN scenario', 2: 'Exit program.'}

    def __init__(self, state = 'options'):
        self.state = state
        self.gannMan = GannMan()

    def options_menu(self):
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


    def input_run_scenario_menu(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.headerText)
        print("--- INPUT SCENARIO ---")
        print("Specify scenario parameters ...")
        # Create parameters
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
        # Run parameters
        selectEpochs = input("Epochs: ")
        selectValInt = input("Validation interval: ")
        selectBestK = input("Best K: ")
        selectMapBatchSize = input("Map batch size: ")
        selectMapLayers = input("Map layers (indices): ")
        selectMapDendrograms = input("Map dendograms (indices): ")
        selectDisplayWeights = input("Weights to be visualized: ")
        selectDisplayBiases = input("Biases to be visualized: ")

        # Create the gann object
        self.gannMan.create_gann(name, networkDims, hiddenActivationFunc, outputActivationFunc,
                                 lossFunc, optimizer, momentumFrac, learningRate, weightInit, dataSource, dataSourceParas,
                                 caseFrac, valFrac, testFrac, miniBatchSize)

        # Run: train and test the gann
        if selectBestK == '': selectBestK = None
        self.gannMan.run_gann(epochs = int(selectEpochs), showInterval = None,
                validationInterval = int(selectValInt), bestk = int(selectBestK), mapBatchSize = selectMapBatchSize,
                mapLayers = selectMapLayers, mapDendrograms = selectMapDendrograms, displayWeights = selectDisplayWeights,
                displayBiases = selectDisplayBiases)


        # When finished, reset the gann man
        del self.gannMan
        self.gannMan = GannMan()

        waitForExit = input("\nPRESS ENTER TO GO BACK MENU..")
        self.state = "options"

    def load_run_scenario_menu(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.headerText)
        print("--- LOAD BEST PARAMETER SCENARIO --- \n")
        print("Available datasources:")
        print("> bitcounter\n> parity\n> segment\n> autoencoder\n> dense_autoencoder\n> MNIST\n> wine\n> yeast\n> glass\n> hackers")

        selectDataSource = input("\nSelect data source: ")
        self.gannMan.do_gann_from_config(selectDataSource)

        waitForExit = input("\n[Press enter to return to the main menu..]")
        self.state = "options"

    def start_ui(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        exit = False
        while(not exit):
            if(self.state == 'options'): self.options_menu()
            if(self.state == 'inputRunScenario'): self.input_run_scenario_menu()
            if(self.state == 'loadRunScenario'): self.load_run_scenario_menu()
            if(self.state == "exit"):
                exit = True
                print("\nExiting program..")
                time.sleep(1.2)





if __name__ == '__main__':
    ui = GannManUi()
    #ui.read_scenario_menu()
    #ui.select_created_scenario()
    #ui.load_best_param_scenario()
    ui.start_ui()# doesn't work with debugging, instead run the function you want to debuf directly
    #ex: read_scenario_menu() to experiment with different networks parameters