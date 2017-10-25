import os
import glob
import time
from msvcrt import getwch
from gannman import *

"""
Class: GannManUi
A simple user-interface for running different architectures of neural networks that lets the user
specify all relevant scenario parameters.

Dependencies: gannman
"""
class GannManUi:
    state = None                                                                        # Which menu/submenu the program is currently running
    gannMan =                                                                           # A gann manager that manages the user inputs
    stateDict = {0: 'inputRunScenario', 1: 'loadRunScenario', 2: 'exit'}                # Submenus
    menuIndexPointer = 0                                                                # The graphical pointer index

    # Text/graphical stuff
    pointer = '>>'
    headerText = '### GANN MANAGER UI - PROJECT 2 DEMO ###\n'
    initMenuOptions = { 0: 'INPUT & RUN scenario', 1: 'LOAD & RUN scenario', 2: 'Exit program.'}

    def __init__(self, state = 'options'):
        self.state = state
        self.gannMan = GannMan()

    # Description: Launches the menu and it's state machine.
    # Input: None
    # Output: None
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
                time.sleep(0.6)

    # Description: Enters the main menu of the program where one can select by using the w and s keys on the keyboard to
    #           navigate, and enter to select an option. There are currently two options: input a scenario/architecture
    #           directly to the interface or by choosing a config file which stores all the parameters.
    # Input: None
    # Output: None
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

    # Description: Starts a menu which lets the user input the full specification of a scenario/architecture,
    #           then passses this information to the gann manager which runs the scenario.
    # Input: None
    # Output: None
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
        optimizer = input("Optimizer (gradient_descent momentum, adam, adagrad): ")
        if optimizer == 'adam' or optimizer == "adagrad" or optimizer == 'momentum':
            optimizerParams = input("Optmizer params (epsilon, accumulator, momentum):")
        else:
            optimizerParams = None
        learningRate = input("Learning rate <0, 1>: ")
        weightInitType = input("Weight initializing method (normalized, uniform): ")
        if weightInitType == 'uniform':
            weightInit = input("Initial weight range (or scaled): ")
        else:
            weightInit = None
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

        confirmation = input("Abort parameters? [y]: ")
        if confirmation == 'y': pass
        else:
            # Create the gann object
            self.gannMan.create_gann(name, networkDims, hiddenActivationFunc, outputActivationFunc,
                                     lossFunc, optimizer, optimizerParams, learningRate, weightInitType, weightInit, dataSource, dataSourceParas,
                                     caseFrac, valFrac, testFrac, miniBatchSize)

            # Run: train and test the gann
            self.gannMan.run_gann(epochs = int(selectEpochs), showInterval = None,
                    validationInterval = int(selectValInt), bestK = selectBestK, mapBatchSize = selectMapBatchSize,
                    mapLayers = selectMapLayers, mapDendrograms = selectMapDendrograms, displayWeights = selectDisplayWeights,
                    displayBiases = selectDisplayBiases)


            # When finished, reset the gann man
            del self.gannMan
            self.gannMan = GannMan()

        waitForExit = input("\nPRESS ENTER TO GO BACK MENU..")
        self.state = "options"

    # Description: Starts a menu which lets the user select a predefined config file which stores all the
    #           parameters for a scenario, then passes the selection to the gann manager which
    #           reads the file and runs the scenario
    # Input: None
    # Output: None
    def load_run_scenario_menu(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.headerText)
        print("--- LOAD BEST PARAMETER SCENARIO --- \n")
        print("Supported config formats: .txt\n")
        print("Available scenarios:\n")
        for root, dirs, files in os.walk("best_param_networks"):
            for file in files:
                if file.endswith('.txt'):
                    print(file)

        selectDataSource = input("\nSelect data source: ")
        confirmation = input("Abort data source selection [y]: ")
        if confirmation == 'y': pass
        else:
            self.gannMan.do_gann_from_config(selectDataSource)

            # When finished, reset the gann man
            del self.gannMan
            self.gannMan = GannMan()

        waitForExit = input("\n[Press enter to return to the main menu..]")
        self.state = "options"




if __name__ == '__main__':
    ui = GannManUi()
    ui.start_ui()