import numpy
import sys
from RushHourGame import *



def main(argv):
    print('The arguments were: ', argv)
    boardHeight = int(argv[0])
    boardLength = int(argv[1])
    gameConfigFile = argv[2]

    # Initialize a game with the config
    Game = RushHourGame(boardHeight, boardLength)
    Game.loadGameConfiguration(gameConfigFile)

    # Display the board
    Game.displayGame()



    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
