import sys
import numpy as np
from termcolor import colored, cprint
import colorama
from string import *


class NonogramCspNode:
    # Internal variables
    rows = None
    cols = None

    # CSP variables
    rowVariables = None
    colVariables = None
    constraints = None

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
        self.rows = None
        self.cols = None

        self.rowVariables = []
        self.colVariables = []
        self.constraints = []

        self.start = None
        self.goal = None
        self.state = None
        self.parent = None
        self.kids = None
        self.g = 0
        self.f = 0
        self.h = 0

# Description: Loads the configuration for the nonogram and builds the variables domain.
# Input: text document with a nonogram config, fileName
# Output: None
    def load_nonogram_configuration(self, fileName):
        with open('../nonograms_configurations/' + fileName + '.txt', 'r') as f:
            # Read nonogram size
            firstLine = f.readline().split(" ")
            self.cols, self.rows = [int(x) for x in firstLine]

            for rowNumber in range(0, self.rows):
                # Read row specifications from file
                rowSpec = f.readline().split(" ")
                rowSpec = [int(x) for x in rowSpec]

                # Translate to a compositions-problem
                minTotalSegmentLength = sum(rowSpec) + len(rowSpec) - 1
                maxSegmentMoves = self.cols - minTotalSegmentLength
                compositionRestrictionLength = (len(rowSpec) + 1)

                # Initialize storage and initial composition, and find all compositions
                initialComposition = [0] * compositionRestrictionLength
                initialComposition[0] = maxSegmentMoves
                rowCompositions = [initialComposition]
                if(initialComposition[0] != 0):
                    self.find_all_weak_compositions(rowCompositions, initialComposition,
                                                    maxSegmentMoves, compositionRestrictionLength)

                # Translate from compositions to a row variables domain
                rowDomain = self.compositions_to_variable_values(rowCompositions, rowSpec)
                self.rowVariables.append(VariableInstance('row', rowNumber, rowDomain))

            for colNumber in range(0, self.cols):
                # Read column specifications from file
                colSpec = f.readline().split(" ")
                colSpec = [int(x) for x in colSpec]

                # Translate to a compositions-problem
                minTotalSegmentLength = sum(colSpec) + len(colSpec) - 1
                maxSegmentMoves = self.rows - minTotalSegmentLength
                compositionRestrictionLength = (len(colSpec) + 1)

                # Initialize storage and initial composition, and find all compositions
                initialComposition = [0] * compositionRestrictionLength
                initialComposition[0] = maxSegmentMoves
                colCompositions = [initialComposition]
                if (initialComposition[0] != 0):
                    self.find_all_weak_compositions(colCompositions, initialComposition,
                                                    maxSegmentMoves, compositionRestrictionLength)

                # Translate from compositions to a column variables domain
                colDomain = self.compositions_to_variable_values(colCompositions, colSpec)
                self.colVariables.append(VariableInstance('col', colNumber, colDomain))

# Description:
# Input:
# Output:
    def create_all_constraints(self):
        for row in self.rowVariables:
            for col in self.colVariables:
                constraintInstance = ConstraintInstance(row, col)
                self.constraints.append(constraintInstance)


# Description: Finds all possible (restricted) weak compositions of the number n with k parts.
#             Continuously appends the compositions to the storage.
# Input: storage compositionsAccumulator, parentNode parentComposition, number n, parts k
# Output: 0
    def find_all_weak_compositions(self, compositionsAccumulator, parentComposition, n, k):
        newCompositions = []
        for i in range(1, k):
            tempComposition = list(parentComposition)
            for j in range(0, k):
                if (j == 0):
                    tempComposition[j] -= 1
                if (j == i):
                    tempComposition[j] += 1
            newCompositions.append(tempComposition)

        for composition in newCompositions:
            if composition not in compositionsAccumulator:
                compositionsAccumulator.append(composition)
                if(composition[0] != 0):
                    self.find_all_weak_compositions(compositionsAccumulator, composition, n, k)

# Description:
# Input:
# Output:
    def compositions_to_variable_values(self, compositions, variableSpec):
        values = []
        for composition in compositions:
            value = []
            for i in range(0, len(composition)):
                if (i != 0 and i != len(composition) - 1):
                    value += ([0] * (composition[i] + 1))
                else:
                    value += ([0] * composition[i])
                if (i < len(composition) - 1):
                    value += ([1] * variableSpec[i])
            values.append(value)
        return values

# Description:
# Input:
# Output:
class ConstraintInstance:
    rowVariableInstance = None
    colVariableInstance = None

    def __init__(self, rowVariableInstance, colVariableInstance):
        self.rowVariableInstance = rowVariableInstance
        self.colVariableInstance = colVariableInstance

    def satisfied(self, focalVariable):
        if(focalVariable == 'col'):
            focalVariable = self.colVariableInstance
            nonFocalVariable = self.rowVariableInstance
        elif(focalVariable == 'row'):
            focalVariable = self.rowVariableInstance
            nonFocalVariable = self.colVariableInstance
        else:
            raise AttributeError

# Description:
# Input:
# Output:
class VariableInstance:
    type = None
    number = None
    domain = None

    def __init__(self, type, number, domain):
        self.type = type
        self.number = number
        self.domain = domain

