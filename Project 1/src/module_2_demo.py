import numpy
import sys
import time
from termcolor import colored
import colorama
from nonogram_csp import *
from astar_search import *
from csp_solver import *
from gac_and_astar import *

def module_2(argv):
    # Initialize color prints
    colorama.init()

    # Show arguments given to the script
    print('[MAIN]: Script arguments: ', argv)
    gameConfigFile, searchType, displayMode = argv[0], argv[1], argv[2]
    if (displayMode == "display"):
        displayMode = True
    else:
        displayMode = False
    print('')

    # Create a nonogram and a solver
    cspSolver = GAC()
    initialNonogramNode = NonogramCspNode(cspSolver = cspSolver)

    # Initialize the csp and solver
    startLoadConfigTime = time.clock()
    initialNonogramNode.load_nonogram_configuration(gameConfigFile)
    initialNonogramNode.create_all_constraints()
    endLoadConfigTime = time.clock()
    cspSolver.initialization(cspConstraints = initialNonogramNode.constraints)
    print('[MAIN]: Loading the config took: ', endLoadConfigTime - startLoadConfigTime, ' seconds.')

    # Find total domain size before using the solver
    oldTotalSize = initialNonogramNode.get_total_domain_size()

    # Run the csp-solver on the csp with domain filtering
    validReduction = cspSolver.domain_filtering_loop()

    # Find total domain size before using the solver
    newTotalSize = initialNonogramNode.get_total_domain_size()

    # Create and initialize an astar-search
    initialNonogramNode.state = initialNonogramNode.get_state_identifier()
    astarSearch = AStar(searchType = searchType, startSearchNode = initialNonogramNode, displayMode = displayMode)

    # Run the GAC_A* algorithm
    gacAstarSolver = GacAstar(initialNonogramNode, cspSolver, astarSearch)
    gacAstarSolver.run(validReduction)



    return 0




if __name__ == '__main__':
    module_2(sys.argv[1:])