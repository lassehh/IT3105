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
    boardHeight, boardLength, gameConfigFile = int(argv[0]), int(argv[1]), argv[2]
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
    initialGameNode.display_game_board()

    # Run the AStar algorithm on all 3 types of search and display the result
    searchTypes = ["BFS", "DFS", "BestFS"]
    for searchType in searchTypes:
        print('\n\n')
        print("[MAIN]: Starting a new search for the solution to \"" + gameConfigFile + "\" with " + searchType + " search type.")
        AStarSearchObject = AStar(searchType = searchType, startSearchNode = initialGameNode, displayMode = displayMode)
        solutionNode, numberOfMovesToSolution, searchNodesGenerated = AStarSearchObject.best_first_search()
        print("[MAIN]: With " + searchType + " search, the solution includes:")
        print("- " + str(numberOfMovesToSolution) + " moves")
        print("- " + str(AStarSearchObject.searchNodesGenerated) + " nodes generated")

    return 0


if __name__ == '__main__':
    module_1(sys.argv[1:])


