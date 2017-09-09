import numpy
import sys
from termcolor import colored
import colorama
from nonogram import *
from a_star_search import *

def module_2(argv):
    # Initialize color prints
    colorama.init()

    # Show arguments given to the script
    print('[MAIN]: Script arguments: ', argv)
    gameConfigFile = argv[0]
    print('')

    # Initialize a nonogram with the config
    initialNonogramNode = NonogramNode()
    initialNonogramNode.load_nonogram_configuration(gameConfigFile)










if __name__ == '__main__':
    module_2(sys.argv[1:])