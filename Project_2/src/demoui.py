import sys
import os
import tflowtools as tf
#import gann
import time
import getchar
import ptyprocess



class ui:
    state = None
    getch = None
    headerText = '--- DEMO UI PROJECT 2 ---'
    initMenuOptions = {
        0: '1. Input scenario parameters',
        1: '2. Select a datasource and build & run a network with the best known parameters'}

    def __init__(self, state = 'init',):
        self.state = state
        self.getchFunc = getchar._Getch()

    def init_menu(self):
        optionSelected = False
        menuIndexPointer = 0
        pointer = '>>'
        while(not optionSelected):
            print(self.headerText)
            for i in range(0,len(self.initMenuOptions)):
                if i == menuIndexPointer:
                    print(pointer + '\t' + self.initMenuOptions[i])
                else:
                    print('\t' + self.initMenuOptions[i])
            character = self.getchFunc()
            if character == 'w': menuIndexPointer = (menuIndexPointer + 1) % 2
            elif character == 's': menuIndexPointer = (menuIndexPointer - 1) % 2
            os.system('cls' if os.name == 'nt' else 'clear')
            time.sleep(0.01)



    def read_scenario_menu(self):
        pass

    def select_datasource_menu(self):
        pass


    def run(self):
        while(1):
            if(self.state == 'init'):
                self.state = self.init_menu()
            elif(self.state == 'readScenario'):
                self.state = self.read_scenario_menu()





if __name__ == '__main__':
    ui = ui()
    ui.run()