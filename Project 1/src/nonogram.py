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
        #self.VI.append(VariableInstance(0))
        with open('../nonograms_configurations/' + fileName + '.txt', 'r') as f:
            firstLine = f.readline().split(" ")
            self.cols, self.rows = [int(x) for x in firstLine]

            for row in range(0, self.rows):
                rowSpec = f.readline().split(" ")
                rowSpec = [int(x) for x in rowSpec]

                minTotalSegmentLength = sum(rowSpec) + len(rowSpec) - 1
                maxSegmentMoves = self.cols - minTotalSegmentLength
                rowCompositions = []
                self.findAllWeakCompositions2(rowCompositions, maxSegmentMoves, len(rowSpec) + 1)

                rowDomain = []
                for composition in rowCompositions:
                    rowValue = []
                    for i in range(0,len(composition)):
                        if(i != 0 and i != len(composition) - 1):
                            rowValue += ([0] * (composition[i] + 1))
                        else:
                            rowValue += ([0] * composition[i])
                        if(i < len(composition) - 1):
                            rowValue += ([1] * rowSpec[i])

                    rowDomain.append(rowValue)
                self.rowVariables.append(rowDomain)

            for col in range(0, self.cols):
                colSpec = f.readline().split(" ")
                colSpec = [int(x) for x in colSpec]

                minTotalSegmentLength = sum(colSpec) + len(colSpec) - 1
                maxSegmentMoves = self.rows - minTotalSegmentLength
                colCompositions = []
                self.findAllWeakCompositions2(colCompositions, maxSegmentMoves, len(colSpec) + 1)

                colDomain = []
                for composition in colCompositions:
                    colValue = []
                    for i in range(0, len(composition)):
                        if (i != 0 and i != len(composition) - 1):
                            colValue += ([0] * (composition[i] + 1))
                        else:
                            colValue += ([0] * composition[i])
                        if (i < len(composition) - 1):
                            colValue += ([1] * colSpec[i])

                    colDomain.append(colValue)
                self.colVariables.append(colDomain)


    def findAllWeakCompositions2(self, compositionsAccumulator, n, k):
        reductionIndex = 0
        initialComposition = [0] * k
        initialComposition[0] = n
        self.generateNewCompositions2(compositionsAccumulator, initialComposition, reductionIndex, n, k)

    def generateNewCompositions2(self, compositionsAccumulator, parentComposition, reductionIndex, n, k):

        if parentComposition not in compositionsAccumulator:
            compositionsAccumulator.append(parentComposition)

        if (parentComposition[reductionIndex] == 0):
            return 0

        else:
            newCompostions = []
            for i in range(0, k):
                if (i == reductionIndex):
                    continue
                tempComposition = list(parentComposition)
                for j in range(0, k):
                    if (j == reductionIndex):
                        tempComposition[j] -= 1
                    if (j == i):
                        tempComposition[j] += 1
                newCompostions.append(tempComposition)

            for composition in newCompostions:
                if composition not in compositionsAccumulator:
                    self.generateNewCompositions2(compositionsAccumulator, composition, reductionIndex, n, k)
            return 0


        ############
# Description: Finds all possible (restricted) weak compositions of the number n with k parts
# Input: number n, parts k
# Output: Variable length list with lists of length k that contains all the possible compositions for n, k
    def findAllWeakCompositions(self, compositionsAccumulator, n, k):
        for reductionIndex in range(0, k):
            initialComposition = []
            for j in range(0, k):
                if(reductionIndex == j):
                    initialComposition.append(n)
                else:
                    initialComposition.append(0)
            self.generateNewCompositions(compositionsAccumulator, initialComposition, reductionIndex, n, k)

############
# Description: Generates (k - 1) new compositions from parentComposition through the reduction index and stores them in
#              compostionAccumulator
# Input:
#  - the storage for all the compositions, compositionsAccumulaator
#  - root node composition, parentComposition
#  - which index to reduce and to generate new compostions from, reductionIndex
#  - how many parts the composition must contain, k
# Output: 0 (nothing), terminates when the diffeence between largest and second largest element of the composition
#       is less than 2 (i.e 1).
    def generateNewCompositions(self, compositionsAccumulator, parentComposition, reductionIndex, n, k):
        parentCompositionCopy = list(parentComposition)

        largestElement = sorted(parentCompositionCopy)[-1]
        secondLargestElement = sorted(parentCompositionCopy)[-2]

        if parentComposition not in compositionsAccumulator:
            compositionsAccumulator.append(parentComposition)

        if(largestElement - secondLargestElement < 2 and n % k > 0):
            return 0
        elif(largestElement - secondLargestElement < 1 and n & k == 0):
            return 0
        else:
            newCompostions = []
            for i in range(0, k):
                if (i == reductionIndex):
                    continue
                tempComposition = list(parentComposition)
                for j in range(0, k):
                    if(j == reductionIndex):
                        tempComposition[j] -= 1
                    if(j == i):
                        tempComposition[j] += 1
                newCompostions.append(tempComposition)


            for composition in newCompostions:
                if composition not in compositionsAccumulator:
                    newReductionIndex = max(range(len(composition)), key = composition.__getitem__)
                    self.generateNewCompositions(compositionsAccumulator, composition, newReductionIndex, n, k)
            return 0

