import numpy
import sys
import time
from termcolor import colored
import colorama
from nonogram_csp import *
from a_star_search import *

def module_2(argv):
    # Initialize color prints
    colorama.init()

    # Show arguments given to the script
    print('[MAIN]: Script arguments: ', argv)
    gameConfigFile = argv[0]
    print('')

    # Initialize a nonogram with the config
    initialNonogramNode = NonogramCspNode()
    startLoadConfigTime = time.clock()
    initialNonogramNode.load_nonogram_configuration(gameConfigFile)
    print('[MAIN]: Loading the config took: ', time.clock() - startLoadConfigTime, ' seconds.')

    # Create all problem constraints
    initialNonogramNode.create_all_constraints()


    #TESTING
    #initialCompostions = initialNonogramNode.findAllWeakCompositions(4, 3)


    return 0










if __name__ == '__main__':
    module_2(sys.argv[1:])