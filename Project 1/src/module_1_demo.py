import numpy
import sys
from termcolor import colored
import colorama
from rush_hour import *
from astar_search import *

def module_1(argv):
    # Initialize color prints
    colorama.init()

    # Show arguments given to the script
    print('[MAIN]: Script arguments: ', argv)
    boardHeight, boardLength, searchType, gameConfigFile = int(argv[0]), int(argv[1]), argv[2], argv[3]
    print('')

    # Ask user for script options
    userInputDisplayOption = input("[MAIN]: Activate display mode (y/n)? ")
    if(userInputDisplayOption == "y"):
        displayMode = True
    else:
        displayMode = False
    print('')

    # Initialize a game with the config
    initialGameNode = RushHourGameNode(boardHeight = boardHeight, boardLength = boardLength)
    initialGameNode.load_game_configuration(gameConfigFile)

    # Display the board
    print("[MAIN]: Board: " + gameConfigFile)
    initialGameNode.update_game_board()
    initialGameNode.display_node()

    # Run the AStar algorithm on all 3 types of search and display the result
    print('\n\n')
    print("[MAIN]: Starting a new search for the solution to \"" + gameConfigFile + "\" with " + searchType + " search type.")
    AStarSearchObject = AStar(searchType = searchType, startSearchNode = initialGameNode, displayMode = displayMode)
    solutionNode, numberOfMovesToSolution, searchNodesGenerated, _ = AStarSearchObject.best_first_search()
    print("[MAIN]: With " + searchType + " search, the solution includes:")
    print("- " + str(numberOfMovesToSolution) + " moves")
    print("- " + str(AStarSearchObject.searchNodesGenerated) + " nodes generated")
    print("[MAIN]: The solution state is:")
    solutionNode.display_node()
    
    return 0


if __name__ == '__main__':
    module_1(sys.argv[1:])


