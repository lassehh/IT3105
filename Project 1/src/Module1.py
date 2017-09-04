import numpy
import sys
from termcolor import colored
import colorama
from RushHourGame import *
from GenericAStar import *



def module_1(argv):
    colorama.init()

    print('Script arguments: ')
    print(argv)
    boardHeight = int(argv[0])
    boardLength = int(argv[1])
    gameConfigFile = argv[2]
    print('')
    #searchType = argv[3]

    # Initialize a game with the config
    initialGameNode = RushHourGameNode(boardHeight = boardHeight, boardLength = boardLength)
    initialGameNode.load_game_configuration(gameConfigFile)

    # Display the board
    print("Board: " + gameConfigFile)
    initialGameNode.update_game_board()
    initialGameNode.display_game_board()

    # Run the AStar algorithm on all 3 types of search and display the result
    searchTypes = ["BFS", "DFS", "BestFS"]
    for searchType in searchTypes:
        AStarSearch = AStar(searchType, initialGameNode)
        solutionNode = AStarSearch.best_first_search()
        numberOfMoves = AStarSearch.get_number_of_moves(solutionNode)
        print("With " + searchType + " search, the solution includes:")
        print("- " + str(numberOfMoves) + " moves")
        print("- " + str(AStarSearch.searchNodesGenerated) + " nodes generated")

    noob = 0
    return 0


if __name__ == '__main__':
    module_1(sys.argv[1:])


