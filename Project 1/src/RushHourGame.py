import sys
import numpy as np
from string import *

class RushHourGame:
    gameBoard = None
    gameBoardSize = []
    numberOfVehicles = 0
    cars = []
    trucks = []


    def __init__(self, boardLength, boardHeight):
        self.gameBoard = np.zeros((boardLength, boardHeight))
        self.gameBoardSize = (boardLength,boardHeight)
        self.numberOfVehicles = 0
        self.cars = []
        self.trucks = []

    def loadGameConfiguration(self, fileName):
        with open('../game_configurations/' + fileName + '.txt', 'r') as f:
            for line in f:
                line = line.strip("\n")
                currentLine = line.split(",")
                currentLine = [int(x) for x in currentLine]
                currentLine = [self.numberOfVehicles] + currentLine
                vehicleType = currentLine[-1]
                if(vehicleType == 2):
                    self.cars.append(currentLine)
                elif(vehicleType == 3):
                    self.trucks.append(currentLine)
                else:
                    raise AttributeError
                self.numberOfVehicles += 1

    def displayGame(self):
        boardDisplay = np.zeros(shape = self.gameBoardSize, dtype = str)
        boardDisplay[:] = 'x'
        for car in self.cars:
            (number, orientation, xPos, yPos, size)  = car[0], car[1], car[2], car[3], car[4]
            if(orientation == 0):
                boardDisplay[yPos, xPos:xPos+size] = number
            elif(orientation == 1):
                boardDisplay[yPos:yPos + size, xPos] = number
        for truck in self.trucks:
            (number, orientation, xPos, yPos, size) = truck[0], truck[1], truck[2], truck[3], truck[4]
            if (orientation == 0):
                boardDisplay[yPos, xPos:xPos + size] = number
            elif (orientation == 1):
                boardDisplay[yPos:yPos + size, xPos] = number

        # Display the board on screen
        for row in boardDisplay:
            for element in row:
                print(element + ' ', end='')
            print('')

