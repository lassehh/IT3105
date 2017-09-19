import sys
import time
import colorama
from nonogram_csp import NonogramCspNode
from astar_search import AStar
from csp_solver import GAC
from gac_and_astar import GacAstar

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
    initialNonogramNode = NonogramCspNode()

    # Generate the initial state, where each variable has its full domain
    startTime = time.clock()
    initialNonogramNode.load_nonogram_configuration(gameConfigFile)
    initialNonogramNode.create_all_constraints()
    endTime = time.clock()

    # Refine the nonogram node by running initialization and domain_filtering_loop on the csp
    cspSolver.initialization(problemObject = initialNonogramNode)
    print('[MAIN]: Loading the config took: ', endTime - startTime, ' seconds.')

    # Run the csp-solver on the csp with domain filtering
    startTime = time.clock()
    validReduction = cspSolver.domain_filtering_loop()
    endTime = time.clock()
    print('[MAIN]: Domain-filtering the nonogram took: ', endTime - startTime, ' seconds.')

    # Create and initialize an astar-search
    initialNonogramNode.state = initialNonogramNode.get_state_identifier()
    astarSearch = AStar(searchType = searchType, startSearchNode = initialNonogramNode, displayMode = displayMode)

    # Display the result after running the csp solver
    print('[MAIN]: Nonogram after domain filtering: ')
    initialNonogramNode.display_node()

    # Run the GAC_A* algorithm (if the refined nonogram node is neither a contradictory state nor a solution)
    gacAstarSolver = GacAstar(initialNonogramNode, astarSearch)
    solution, numberOfMovesToSolution, searchNodesGenerated, searchNodesExpanded = gacAstarSolver.run(validReduction)


    # Print the solved nonogram
    print('[MAIN]: Solution of "' + gameConfigFile + '":')
    solution.display_node()
    if numberOfMovesToSolution > -1:
        print("[MAIN]: With " + searchType + " search, the solution includes:")
        print("- " + str(numberOfMovesToSolution) + " moves")
        print("- " + str(searchNodesGenerated) + " nodes generated")
        print("- " + str(searchNodesExpanded) + " nodes expanded")
    else:
        print("[MAIN]: A* with " + searchType + " was not necessary.")

    return 0




if __name__ == '__main__':
    module_2(sys.argv[1:])