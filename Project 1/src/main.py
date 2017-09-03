import numpy
import sys
from RushHourGame import *
from GenericAStar import *



def main(argv):
    print('The arguments were: ', argv)
    boardHeight = int(argv[0])
    boardLength = int(argv[1])
    gameConfigFile = argv[2]

    # Initialize a game with the config
    initialGameNode = RushHourGameNode()
    initialGameNode.load_game_configuration(gameConfigFile)

    # Display the board
    initialGameNode.update_game_board()
    initialGameNode.display_game_board()

    AStarSearch = AStar("DFS", initialGameNode)
    solution = AStarSearch.best_first_search()

    noob = 0




    #Test
    #noobVar1 = initialGameNode.isGoal()
    #initialGameNode.vehicles[0] = [0,0,4,2,2]
    #noobVar2 = initialGameNode.isGoal()
    #noobVar3 = initialGameNode.get_state_identifier(initialGameNode.vehicles)
    # noobs = initialGameNode.generate_successors()
    # for noob in noobs:
    #     noob.update_game_board()
    #     noob.display_game_board()
    #     print('\n')

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
