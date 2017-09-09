import sys
import numpy as np
from termcolor import colored, cprint
import colorama
from string import *

class NonogramNode:
    # Internal variables
    rows = None
    cols = None

    # CSP variables
    rowVariables = None
    colVariables = None

    # A* variables
    start = None
    goal = None
    state = None
    parent = None
    kids = None
    g = 0
    f = 0
    h = 0

    def __init__(self):
        self.VI = []

        self.start = None
        self.goal = None
        self.state = None
        self.parent = None
        self.kids = None
        self.g = 0
        self.f = 0
        self.h = 0


    def load_nonogram_configuration(self, fileName):
        #self.VI.append(VariableInstance(0))
        with open('../nonograms_configurations/' + fileName + '.txt', 'r') as f:
            firstLine = f.readline().split(" ")
            self.cols, self.rows = [int(x) for x in firstLine]

            for row in range(0, self.rows):
                rowSpec = f.readline().split(" ")
                rowSpec = [int(x) for x in rowSpec]

                minTotalSegmentLength = sum(rowSpec) + len(rowSpec) - 1
                maxSegmentMoves = self.rows - minTotalSegmentLength



            for i in range(0, self.cols):
                colSpec = f.readline().split(" ")
                colSpec = [int(x) for x in colSpec]

                minTotalSegmentLength = sum(colSpec) + len(colSpec) - 1
                maxSegmentMoves = self.rows - minTotalSegmentLength

            noob = 0


    #def find_variable_domains(self, tempDomain):
