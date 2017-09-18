import sys
import numpy as np
from termcolor import colored, cprint
import colorama
from string import *
import copy
from csp_solver import *

#
class ConstraintInstance:
    # References to the variables involved in the constraint
    rowNumber = None
    colNumber = None

    def __init__(self, rowVariableInstance, colVariableInstance):
        self.rowNumber = rowVariableInstance
        self.colNumber = colVariableInstance

    def get_all_variables(self, problemObject):
        allVariables = [problemObject.rowVariables[self.rowNumber], problemObject.colVariables[self.colNumber]]
        return allVariables

    def get_non_focal_variable(self, focalVariable, problemObject):
        if (focalVariable.number == self.rowNumber and focalVariable.type == 'row'):
            nonFocalVariable = problemObject.colVariables[self.colNumber]
        elif (focalVariable.number == self.colNumber and focalVariable.type == 'col'):
            nonFocalVariable = problemObject.rowVariables[self.rowNumber]
        else:
            raise AssertionError
        return nonFocalVariable

    def satisfied(self, value, domainValue, focalVariable):
        if(value == domainValue[focalVariable.number]):
            return True
        else:
            return False

    def get_pruning_values(self, focalVariable, problemObject):
        nonFocalVariable = self.get_non_focal_variable(focalVariable, problemObject)

        if all(domainValue[nonFocalVariable.number] == 0 for domainValue in focalVariable.domain):
            return [0]
        elif all(domainValue[nonFocalVariable.number] == 1  for domainValue in focalVariable.domain):
            return [1]
        else:
            return []

#
class VariableInstance:
    type = None
    number = None
    domain = None

    def __init__(self, type, number, domain):
        self.type = type
        self.number = number
        self.domain = domain


class NonogramCspNode:
    cspSolver = None

    # Internal variables
    rows = None
    cols = None
    colors = None

    # CSP variables
    rowVariables = None
    colVariables = None
    constraints = None

    # A* variables
    start = None
    #goal = None
    state = None
    parent = None
    kids = None
    g = 0
    f = 0
    h = 0

    def __init__(self, cspSolver):
        self.cspSolver = cspSolver

        self.rows = None
        self.cols = None
        self.colors = {("0"): "white", ("1"): "red"}

        self.rowVariables = []
        self.colVariables = []
        self.constraints = []

        self.start = None
        self.goal = None
        self.state = None
        self.parent = None
        self.kids = []
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
                if(maxSegmentMoves != 0):
                    self.find_all_weak_compositions(rowCompositions, initialComposition,
                                                    maxSegmentMoves, compositionRestrictionLength)

                # Translate from compositions to a row variables domain
                rowDomain = self.compositions_to_variable_values(rowCompositions, rowSpec)
                self.rowVariables.append(VariableInstance('row', rowNumber, rowDomain))

            for colNumber in range(0, self.cols):
                # Read column specifications from file
                colSpec = f.readline().split(" ")
                colSpec = [int(x) for x in colSpec]
                colSpec.reverse() # to get the correct order according to the exercise text

                # Translate to a compositions-problem
                minTotalSegmentLength = sum(colSpec) + len(colSpec) - 1
                maxSegmentMoves = self.rows - minTotalSegmentLength
                compositionRestrictionLength = (len(colSpec) + 1)

                # Initialize storage and initial composition, and find all compositions
                initialComposition = [0] * compositionRestrictionLength
                initialComposition[0] = maxSegmentMoves
                colCompositions = [initialComposition]
                if (maxSegmentMoves != 0):
                    self.find_all_weak_compositions(colCompositions, initialComposition,
                                                    maxSegmentMoves, compositionRestrictionLength)

                # Translate from compositions to a column variables domain
                colDomain = self.compositions_to_variable_values(colCompositions, colSpec)
                self.colVariables.append(VariableInstance('col', colNumber, colDomain))

    def print_solved_nonogram(self):
        rows = list(self.rowVariables)
        rows.reverse()
        for row in rows:
            for values in row.domain:
                for value in values:
                    print(colored((str(value )+ ' '), self.colors[str(value)]), end ='')
            print('')
        print('')

# Description:
# Input:
# Output:
    def create_all_constraints(self):
        for row in self.rowVariables:
            for col in self.colVariables:
                constraintInstance = ConstraintInstance(row.number, col.number)
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
    def is_goal(self):
        totalDomainSize = self.get_total_domain_size()
        # Check if all domain sizes are reduced to one
        if totalDomainSize == (self.rows + self.cols):
            return True
        else:
            return False

    def generate_successors(self):
        successors = []
        variables = self.rowVariables + self.colVariables

        # Find the variable with the smallest domain
        smallestDomainVar = min((variable for variable in variables if len(variable.domain) > 1), key = lambda variable: len(variable.domain))

        # Reduce the variables domain to a singleton for every value and append it has a successor state
        for domainValue in smallestDomainVar.domain:
            # Create a copy to modify, only needs deepcopy of the variables
            cspSolver = GAC()
            nonogramChildNode = NonogramCspNode(cspSolver)
            nonogramChildNode.rowVariables = copy.deepcopy(self.rowVariables)
            nonogramChildNode.colVariables = copy.deepcopy(self.colVariables)
            nonogramChildNode.constraints = self.constraints
            nonogramChildNode.rows = self.rows
            nonogramChildNode.cols = self.cols

            cspSolver.set_problem_ref(nonogramChildNode)

            if(smallestDomainVar.type == 'col'):
                childSmallestDomainVar = nonogramChildNode.colVariables[smallestDomainVar.number]
                childSmallestDomainVar.domain = [domainValue]
            else:
                childSmallestDomainVar = nonogramChildNode.rowVariables[smallestDomainVar.number]
                childSmallestDomainVar.domain = [domainValue]

            #TESTING
            oldSize = nonogramChildNode.get_total_domain_size()
            validReduction = nonogramChildNode.cspSolver.rerun(childSmallestDomainVar)
            newSize = nonogramChildNode.get_total_domain_size()
            #TESTING

            if(validReduction):
                nonogramChildNode.state = nonogramChildNode.get_state_identifier()
                successors.append(nonogramChildNode)
        return successors

    def get_state_identifier(self):
        csp_identifier = ''
        variables = self.rowVariables + self.colVariables
        for variable in variables:
            for vector in variable.domain:
                csp_identifier += ''.join(map(str, vector))
        return csp_identifier

    def get_total_domain_size(self):
        totalDomainSize = 0
        for variable in self.rowVariables:
            totalDomainSize += len(variable.domain)
        for variable in self.colVariables:
            totalDomainSize += len(variable.domain)
        return totalDomainSize

    def get_variable(self, spec):
        type, number = spec
        if type == 'col':
            variable = self.colVariables[number]
        elif type == 'row':
            variable = self.rowVariables[number]
        else:
            raise AssertionError
        return variable

    def calc_h(self):
        estimatedDistanceToGoal = self.get_total_domain_size() - self.cols - self.rows
        if (estimatedDistanceToGoal <= 0):
            self.h = 1
        else:
            self.h = estimatedDistanceToGoal

    def arc_cost(self, childNode):
        domainSizeChild = childNode.get_total_domain_size()
        domainSizeParent = self.get_total_domain_size()
        domainSizeDiff = domainSizeParent - domainSizeChild
        return domainSizeDiff
