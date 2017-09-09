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
                maxSegmentMoves = self.cols - minTotalSegmentLength
                rowDomain = []
                self.findAllWeakCompositions(rowDomain, maxSegmentMoves, len(rowSpec) + 1)




            for i in range(0, self.cols):
                colSpec = f.readline().split(" ")
                colSpec = [int(x) for x in colSpec]

                minTotalSegmentLength = sum(colSpec) + len(colSpec) - 1
                maxSegmentMoves = self.rows - minTotalSegmentLength

            noob = 0


# Description: Finds all possible weak compositions of the number n with k parts
# Input: number n, parts k
# Output: Variable length list with lists of length k
    def findAllWeakCompositions(self, rowDomain, n, k):
        for i in range(0, k):
            initialComposition = []
            for j in range(0, k):
                if(i == j):
                    initialComposition.append(n)
                else:
                    initialComposition.append(0)
            self.partition(rowDomain, initialComposition, i, k)

    def partition(self, rowDomain, parentComposition, reductionIndex, k):
        parentCompositionCopy = list(parentComposition)

        largestElement = sorted(set(parentCompositionCopy))[-1]
        secondLargestElement = sorted(set(parentCompositionCopy))[-2]

        if parentComposition not in rowDomain:
            rowDomain.append(parentComposition)

        if(largestElement - secondLargestElement < 2):
            return 0
        else:
            newCompostions = []
            for i in range(0, k):
                tempComposition = list(parentComposition)
                for j in range(0, k):
                    if(j == reductionIndex):
                        tempComposition[j] -= 1
                    if(j == i):
                        tempComposition[j] += 1
                if (i == reductionIndex):
                    continue
                else:
                    newCompostions.append(tempComposition)


            for composition in newCompostions:
                self.partition(rowDomain, composition, reductionIndex, k)
            return 0

