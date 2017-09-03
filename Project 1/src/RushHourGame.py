import sys
import numpy as np
from string import *



class RushHourGameNode:
    gameBoard = None
    gameBoardSize = []
    numberOfVehicles = 0
    vehicles = []

    start = None
    goal = None

    state = None
    parent = None
    kids = None
    g = 0
    f = 0
    h = 0


    def __init__(self, boardLength = 6, boardHeight = 6, vehicles = [], numberOfVehicles = 0, state = '', goal = (5,2)):
        self.gameBoard = np.zeros((boardLength, boardHeight))
        self.gameBoardSize = (boardLength,boardHeight)
        self.numberOfVehicles = numberOfVehicles
        self.vehicles = vehicles
        self.goal = goal
        self.state = state
        self.parent = None
        self.kids = []
        self.g = 0


    def load_game_configuration(self, fileName):
        with open('../game_configurations/' + fileName + '.txt', 'r') as f:
            for line in f:
                line = line.strip("\n")
                currentLine = line.split(",")
                self.state = self.state + currentLine[1] + currentLine[2]
                currentLine = [int(x) for x in currentLine]
                currentLine = [self.numberOfVehicles] + currentLine
                self.vehicles.append(currentLine)
                self.numberOfVehicles += 1

    def update_game_board(self):
        self.gameBoard = np.zeros(shape = self.gameBoardSize, dtype = str)
        self.gameBoard[:] = 'x'
        for vehicle in self.vehicles:
            (number, orientation, xPos, yPos, size)  = vehicle[0], vehicle[1], vehicle[2], vehicle[3], vehicle[4]
            if(orientation == 0):
                self.gameBoard[yPos, xPos:xPos+size] = number
            elif(orientation == 1):
                self.gameBoard[yPos:yPos + size, xPos] = number

    def display_game_board(self):
        # Display the board on screen
        for row in self.gameBoard:
            for element in row:
                print(element + ' ', end='')
            print('')

    def get_state_identifier(self, vehicles):
        state = ''
        for vehicle in vehicles:
            state = state + str(vehicle[2]) + str(vehicle[3])
        return state

    def calc_h(self):
        self.h = 0
        return 0

    def arc_cost(self):
        return 1

    def is_goal(self):
        objectiveVehicle = self.vehicles[0]
        xPos, yPos, size = objectiveVehicle[2], objectiveVehicle[3], objectiveVehicle[4]
        objectiveVehicleRightMostPos = (xPos + size - 1, yPos)
        if(objectiveVehicleRightMostPos == self.goal):
            return True
        else:
            return False

    def generate_successors(self):
        succesors = []
        for vehicle in self.vehicles:
            (number, orientation, xPos, yPos, size) = vehicle[0], vehicle[1], vehicle[2], vehicle[3], vehicle[4]
            # Horizontal vehicle
            if(orientation == 0):
                # The left square is free and is not outside the game board
                if(xPos > 0 and self.gameBoard[yPos,xPos - 1] == 'x'):
                    newVehicleConfig = list(self.vehicles)
                    newVehicleConfig[number] = [number, orientation, xPos - 1, yPos, size]
                    newGameNode = RushHourGameNode(vehicles = newVehicleConfig, numberOfVehicles = self.numberOfVehicles,
                                                   state = self.get_state_identifier(newVehicleConfig))
                    succesors.append(newGameNode)
                    continue
                # The right square is free and is not outside the game board
                elif(xPos + size - 1 < 5 and self.gameBoard[yPos,xPos + size] == 'x'):
                    newVehicleConfig = list(self.vehicles)
                    newVehicleConfig[number] = [number, orientation, xPos + 1, yPos, size]
                    newGameNode = RushHourGameNode(vehicles=newVehicleConfig, numberOfVehicles=self.numberOfVehicles,
                                                   state=self.get_state_identifier(newVehicleConfig))
                    succesors.append(newGameNode)
                    continue
            # Vertical vehicle
            elif(orientation == 1):
                # The square above is free and is not outside the game board
                if (yPos > 0 and self.gameBoard[yPos - 1, xPos] == 'x'):
                    newVehicleConfig = list(self.vehicles)
                    newVehicleConfig[number] = [number, orientation, xPos, yPos - 1, size]
                    newGameNode = RushHourGameNode(vehicles=newVehicleConfig, numberOfVehicles=self.numberOfVehicles,
                                                   state=self.get_state_identifier(newVehicleConfig))
                    succesors.append(newGameNode)
                    continue
                # The square below is free and is not outside the game board
                elif (yPos + size - 1 < 5 and self.gameBoard[yPos + size, xPos] == 'x'):
                    newVehicleConfig = list(self.vehicles)
                    newVehicleConfig[number] = [number, orientation, xPos, yPos + 1, size]
                    newGameNode = RushHourGameNode(vehicles=newVehicleConfig, numberOfVehicles=self.numberOfVehicles,
                                                   state=self.get_state_identifier(newVehicleConfig))
                    succesors.append(newGameNode)
                    continue
        return succesors


