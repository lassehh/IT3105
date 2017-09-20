import sys
import numpy as np
from termcolor import colored, cprint
import colorama
from string import *


"""
Class: RushHourGameNode
Represents a state in a rush hour game with board and vehicles.
"""
class RushHourGameNode:
    # Internal variables
    gameBoard = None
    gameBoardSize = []
    numberOfVehicles = 0
    vehicles = []
    vehicleColors = None

    # A* variables
    start = None
    goal = None
    state = None
    parent = None
    kids = None
    g = 0
    f = 0
    h = 0


    def __init__(self, boardLength = 6, boardHeight = 6, vehicles = [], numberOfVehicles = 0, state = '', goal = (5,2)):
        self.vehicles = vehicles
        self.vehicleColors = {("0"): "red", ("1"): "yellow", ("2"): "magenta", ("3"): "green", ("4"): "cyan", ("5"): "blue",
                              ("6"): "green", ("7"): "yellow", ("8"): "magenta", ("9"): "green", ("10"): "magenta",
                              ("11"): "cyan", ("12"): "green", ("x"): "white"}
        self.numberOfVehicles = numberOfVehicles
        self.gameBoardSize = (boardLength,boardHeight)
        self.update_game_board()
        self.goal = goal
        self.state = state
        self.parent = None
        self.kids = []
        self.g = 0

    # Description: Loads the configuration of vehicles in a puzzle
    # Input: file holding the configuration info, fileName
    # Output: None
    def load_game_configuration(self, fileName):
        with open('../rush_hour_game_configurations/' + fileName + '.txt', 'r') as f:
            for line in f:
                line = line.strip("\n")
                currentLine = line.split(",")
                self.state = self.state + currentLine[1] + currentLine[2]
                currentLine = [int(x) for x in currentLine]
                currentLine = [self.numberOfVehicles] + currentLine
                self.vehicles.append(currentLine)
                self.numberOfVehicles += 1

    # Description: Draws the game board based on the position of the vehicles
    # Input: None
    # Output: None
    def update_game_board(self):
        self.gameBoard = np.zeros(shape = self.gameBoardSize, dtype = object)
        self.gameBoard[:] = 'x'
        for vehicle in self.vehicles:
            (number, orientation, xPos, yPos, size)  = vehicle[0], vehicle[1], vehicle[2], vehicle[3], vehicle[4]
            if(orientation == 0):
                self.gameBoard[yPos, xPos:xPos+size] = str(number)
            elif(orientation == 1):
                self.gameBoard[yPos:yPos + size, xPos] = str(number)

    # Description: Displays the game board
    # Input: None
    # Output: None
    def display_node(self):
        for row in self.gameBoard:
            for element in row:
                print(colored((element + ' '), self.vehicleColors[str(element)]), end ='')
            print('')
        print('')

    # Description: Returns an unique state identifier for the rush hour game state
    # Input: the vehicles on the board, vehicles
    # Output: The unique state identfier, uniqueStateIdentifier
    def get_state_identifier(self, vehicles):
        uniqueStateIdentifier = ''
        for vehicle in vehicles:
            uniqueStateIdentifier = uniqueStateIdentifier + str(vehicle[2]) + str(vehicle[3])
        return uniqueStateIdentifier

    # Description: Evaluates a game state to estimate how many steps it's away from the solution.
    #              Adds 1 steps for each vehicle blocking car-0, and the distance from car-0 to the goal
    #              square (5,3). Saves the result to self.h. Sets self.h = 1 if it estimates the solution
    #              to be less than 1 step away (to secure an admissable heuristic).
    # Input: None
    # Output: None
    def calc_h(self):
        estimatedMovesToSolution = 0
        objectiveVehicle = self.vehicles[0]
        (xPos, yPos, size) = objectiveVehicle[2], objectiveVehicle[3], objectiveVehicle[4]
        step = 1

        while(xPos + (size - 1) + step < 6):
            if(self.gameBoard[yPos, xPos + (size - 1) + step] != 'x'):
                estimatedMovesToSolution += 1
            step += 1
        estimatedMovesToSolution += step

        if(estimatedMovesToSolution <= 0):
            self.h = 1
        else:
            self.h = estimatedMovesToSolution

    # Description: Returns the cost of moving from one state to another.
    # Input:
    # Output: 1
    def arc_cost(self, childNode):
        return 1

    # Description: Evaluates the current game state to identify if it's the solution
    # Input:
    # Output: True/False
    def is_goal(self):
        objectiveVehicle = self.vehicles[0]
        xPos, yPos, size = objectiveVehicle[2], objectiveVehicle[3], objectiveVehicle[4]
        objectiveVehicleRightMostPos = (xPos + size - 1, yPos)
        if(objectiveVehicleRightMostPos == self.goal):
            return True
        else:
            return False

    # Description: Iterates through every vehicle on the board, finding feasible moves, and for every one feasible
    #              move create a new game node with the new move made on the game board and finally appending this to
    #              the successors list. Horizontal vehicles may move left and right, vertical vehicles up and down.
    #
    #              To move an vehicle the neighboring square must not already be occupied by a vehicle or be outside
    #              the board.
    # Input: None
    # Output: Nodes with the new feasible moves made on them, succesors
    def generate_successors(self):
        successors = []
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
                    successors.append(newGameNode)
                # The right square is free and is not outside the game board
                if (xPos + size - 1 < 5 and self.gameBoard[yPos, xPos + size] == 'x'):
                    newVehicleConfig = list(self.vehicles)
                    newVehicleConfig[number] = [number, orientation, xPos + 1, yPos, size]
                    newGameNode = RushHourGameNode(vehicles=newVehicleConfig, numberOfVehicles=self.numberOfVehicles,
                                                   state=self.get_state_identifier(newVehicleConfig))
                    successors.append(newGameNode)
            # Vertical vehicle
            elif(orientation == 1):
                # The square above is free and is not outside the game board
                if(yPos > 0 and self.gameBoard[yPos - 1, xPos] == 'x'):
                    newVehicleConfig = list(self.vehicles)
                    newVehicleConfig[number] = [number, orientation, xPos, yPos - 1, size]
                    newGameNode = RushHourGameNode(vehicles = newVehicleConfig, numberOfVehicles = self.numberOfVehicles,
                                                   state = self.get_state_identifier(newVehicleConfig))
                    successors.append(newGameNode)
                # The square below is free and is not outside the game board
                if(yPos + size - 1 < 5 and self.gameBoard[yPos + size, xPos] == 'x'):
                    newVehicleConfig = list(self.vehicles)
                    newVehicleConfig[number] = [number, orientation, xPos, yPos + 1, size]
                    newGameNode = RushHourGameNode(vehicles = newVehicleConfig, numberOfVehicles = self.numberOfVehicles,
                                                   state = self.get_state_identifier(newVehicleConfig))
                    successors.append(newGameNode)
        return successors
