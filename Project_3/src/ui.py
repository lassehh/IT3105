import os
import glob
import time
import som
from msvcrt import getwch

"""
Class: UI
"""
class Ui:
    state = None                                                                        # Which menu/submenu the program is currently running
    stateDict = {0: 'testTsp', 1: 'bestTsp', 2: 'testIcp', 3: 'bestIcp', 4: 'exit'}                         # Submenus
    menuIndexPointer = 0                                                                # The graphical pointer index

    # Text/graphical stuff
    pointer = '>>'
    headerText = '### SOM UI - PROJECT 3 DEMO ###\n'
    initMenuOptions = { 0: 'TEST scenarios on TSP', 1: 'RUN BEST scenario on TSP',
                        2: 'TEST scenarios on ICP', 3: 'RUN BEST scenario on ICP', 4: 'Exit program.'}

    def __init__(self, state = 'options'):
        self.state = state

    # Description: Launches the menu and it's state machine.
    # Input: None
    # Output: None
    def start_ui(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        exit = False
        while(not exit):
            if(self.state == 'options'): self.options_menu()
            if(self.state == 'testTsp'): self.test_tsp_interface()
            if(self.state == 'bestTsp'): self.best_tsp_interface()
            if(self.state == 'testIcp'): self.test_icp_interface()
            if(self.state == 'bestIcp'): self.best_icp_interface()
            if(self.state == "exit"):
                exit = True
                print("\nExiting program..")
                time.sleep(0.6)

    # Description:
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


    def best_tsp_interface(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.headerText)
        print("--- LOAD BEST PARAMETER SCENARIO FOR TSP --- \n")
        print("Supported config formats: .txt\n")
        print("Available scenarios:\n")
        for root, dirs, files in os.walk("../configsTSP"):
            for file in files:
                if file.endswith('.txt'):
                    print(file)

        fileName = input("\nSelect data source: ")
        confirmInput = input("Proceed with the chosen parameters[y/n]? ")
        if confirmInput == 'y':
            with open('../configsTSP/' + fileName, 'r') as f:
                for paramLine in f:
                    paramLine = paramLine.strip("\n")
                    paramLine = paramLine.split(",")
                    if (paramLine[0] == ''):
                        continue  # Skip empty lines
                    elif (paramLine[0][0] == '#'):
                        continue  # Skip comments
                    else:
                        paramName = paramLine[0]
                        paramLine.pop(0)
                        if paramName == 'problemNumber': problemNumber = int(paramLine[0])
                        elif paramName == 'plotInterval': plotInterval = int(paramLine[0])
                        elif paramName == 'epochs': epochs = int(paramLine[0])
                        elif paramName == 'sigma_0': sigma_0 = float(paramLine[0])
                        elif paramName == 'tau_sigma': tau_sigma = int(paramLine[0])
                        elif paramName == 'eta_0': eta_0 = float(paramLine[0])
                        elif paramName == 'tau_eta': tau_eta = int(paramLine[0])
                        else: raise AssertionError("Parameter: " + paramName + ", is not a valid parameter name.")
            tspSOM = som.SOM(problemType='TSP', problemArg=problemNumber, plotInterval=plotInterval,
                             epochs=epochs, sigma_0=sigma_0, tau_sigma=tau_sigma, eta_0=eta_0, tau_eta=tau_eta)
            tspSOM.run()


        wait = input("PRESS ENTER TO EXIT TO MAIN MENU")
        self.state = "options"
        pass

    def best_icp_interface(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("--- LOAD BEST PARAMETER SCENARIO FOR ICP --- \n")
        print("Supported config formats: .txt\n")
        print("Available scenarios:\n")
        for root, dirs, files in os.walk("../configsICP"):
            for file in files:
                if file.endswith('.txt'):
                    print(file)

        fileName = input("\nSelect data source: ")
        confirmInput = input("Proceed with the chosen parameters[y/n]? ")
        if confirmInput == 'y':
            with open('../configsICP/' + fileName, 'r') as f:
                for paramLine in f:
                    paramLine = paramLine.strip("\n")
                    paramLine = paramLine.split(",")
                    if (paramLine[0] == ''):
                        continue  # Skip empty lines
                    elif (paramLine[0][0] == '#'):
                        continue  # Skip comments
                    else:
                        paramName = paramLine[0]
                        paramLine.pop(0)
                        if paramName == 'gridSize': gridSize = int(paramLine[0])
                        elif paramName == 'gridSize': gridSize = int(paramLine[0])
                        elif paramName == 'epochs': epochs = int(paramLine[0])
                        elif paramName == 'sigma_0': sigma_0 = float(paramLine[0])
                        elif paramName == 'tau_sigma': tau_sigma = int(paramLine[0])
                        elif paramName == 'eta_0': eta_0 = float(paramLine[0])
                        elif paramName == 'tau_eta': tau_eta = int(paramLine[0])
                        elif paramName == 'plotInterval': plotInterval = int(paramLine[0])
                        elif paramName == 'testInterval': testInterval = int(paramLine[0])
                        elif paramName == 'nmbrOfCases': nmbrOfCases = int(paramLine[0])
                        else: raise AssertionError("Parameter: " + paramName + ", is not a valid parameter name.")
            icpSOM = som.SOM(problemType='ICP', problemArg=None, gridSize=gridSize,
                             epochs=epochs, sigma_0=sigma_0, tau_sigma=tau_sigma, eta_0=eta_0, tau_eta=tau_eta,
                             plotInterval=plotInterval, testInterval=testInterval, fillIn=True, nmbrOfCases=nmbrOfCases)
            icpSOM.run()


        wait = input("PRESS ENTER TO EXIT TO MAIN MENU")
        self.state = "options"


    def test_tsp_interface(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.headerText)
        print("--- TESTING TCP SCENARIO ---")

        problemNumber = input('Choose which problem (number): ')
        epochs = input("Epochs: ")
        eta = input("Initial learning rate: ")
        tauEta = input("Learning rate time constant: ")
        sigma = input("Initial neighbourhood size: ")
        tauSigma = input("Neighbourhood time constant: ")
        plotInterval = input("Plot interval: ")

        time.sleep(0.7)
        confirmInput = input("Proceed with the chosen parameters[y/n]? ")
        if confirmInput == 'y':
            tspSOM = som.SOM(problemType='TSP', problemArg=int(problemNumber),
                         epochs=int(epochs), sigma_0=float(sigma), tau_sigma=float(tauSigma), eta_0=float(eta),
                         tau_eta=float(tauEta), plotInterval=int(plotInterval))

            tspSOM.run()
        wait = input("PRESS ENTER TO EXIT TO MAIN MENU")
        self.state = "options"


    def test_icp_interface(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.headerText)
        print("--- TESTING ICP SCENARIO ---")

        epochs = input("Epochs: ")
        gridSize = input("Grid size: ")
        eta = input("Initial learning rate: ")
        tauEta = input("Learning rate time constant: ")
        sigma = input("Initial neighbourhood size: ")
        tauSigma = input("Neighbourhood time constant: ")
        plotInterval = input("Plot interval: ")
        testInterval = input("Test interval: ")
        nmbrOfCases = input("Number of training cases to use: ")
        fillIn = input("Fill in non-classified nodes wrt neighbours[y/n]: ")
        if fillIn == "y":
            fillIn = True
        else:
            fillIn = False

        time.sleep(0.7)
        confirmInput = input("Proceed with the chosen parameters[y/n]? ")
        if confirmInput == 'y':
            icpSOM = som.SOM(problemType='ICP', problemArg=None, gridSize=int(gridSize), initialWeightRange=(0, 1),
                             epochs=int(epochs), sigma_0=float(sigma), tau_sigma=float(tauSigma), eta_0=float(eta),
                             tau_eta=float(tauEta), plotInterval=int(plotInterval), testInterval=int(testInterval),
                             fillIn=True, nmbrOfCases=int(nmbrOfCases))
            icpSOM.run()

        wait = input("PRESS ENTER TO EXIT TO MAIN MENU")
        self.state = "options"



if __name__ == '__main__':
    ui = Ui()
    ui.start_ui()