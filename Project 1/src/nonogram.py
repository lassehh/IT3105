import sys
import numpy as np
from termcolor import colored, cprint
import colorama
from string import *

#GitKrakenTest!

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
        self.rows = None
        self.cols = None

        self.rowVariables = []
        self.colVariables = []

        self.start = None
        self.goal = None
        self.state = None
        self.parent = None
        self.kids = None
        self.g = 0
        self.f = 0
        self.h = 0


    def load_nonogram_configuration(self, fileName):
        with open('../nonograms_configurations/' + fileName + '.txt', 'r') as f:
            # Read nonogram size
            firstLine = f.readline().split(" ")
            self.cols, self.rows = [int(x) for x in firstLine]

            for row in range(0, self.rows):
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
                self.find_all_weak_compositions(rowCompositions, initialComposition, maxSegmentMoves, compositionRestrictionLength)

                # Translate from compositions to a row variables domain
                rowDomain = self.compositions_to_variable_values(rowCompositions, rowSpec)
                self.rowVariables.append(rowDomain)

            for col in range(0, self.cols):
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
                self.find_all_weak_compositions(colCompositions, initialComposition, maxSegmentMoves, compositionRestrictionLength)

                # Translate from compositions to a column variables domain
                colDomain = self.compositions_to_variable_values(colCompositions, colSpec)
                self.colVariables.append(colDomain)


# Description: Finds all possible (restricted) weak compositions of the number n with k parts.
#             Continuously appends the compositions to the storage.
# Input: storage compositionsAccumulator, parentNode parentComposition, number n, parts k
# Output: 0
    def find_all_weak_compositions(self, compositionsAccumulator, parentComposition, n, k):
        if (parentComposition[0] != 0):
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
                    #if(composition[0] != 0):
                    self.find_all_weak_compositions(compositionsAccumulator, composition, n, k)

# Description:
# Input:
# Output:
    def compositions_to_variable_values(self, compositions, variableSpec):
        values = []
        for composition in compositions:
            colValue = []
            for i in range(0, len(composition)):
                if (i != 0 and i != len(composition) - 1):
                    colValue += ([0] * (composition[i] + 1))
                else:
                    colValue += ([0] * composition[i])
                if (i < len(composition) - 1):
                    colValue += ([1] * variableSpec[i])
            values.append(colValue)
        return values
