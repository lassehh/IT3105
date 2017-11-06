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
    stateDict = {0: 'solveTSP', 1: 'testMNIST', 2: 'testOTHER'}                         # Submenus
    menuIndexPointer = 0                                                                # The graphical pointer index

    # Text/graphical stuff
    pointer = '>>'
    headerText = '### UI - PROJECT 3 DEMO ###\n'
    initMenuOptions = { 0: 'Solve TSP with a SOM', 1: 'Test trained SOM on MNIST cases', 2: 'N/A.'
        , 3: 'Exit program.'}

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
            if(self.state == 'solveTSP'): pass
            if(self.state == 'testMNIST'): pass
            if(self.state == 'testOTHER'): pass
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


    # Description:
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
            pass
            #TODO


        waitForExit = input("\n[Press enter to return to the main menu..]")
        self.state = "options"




if __name__ == '__main__':
    ui = Ui()
    ui.start_ui()