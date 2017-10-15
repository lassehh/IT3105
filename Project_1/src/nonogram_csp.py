from termcolor import colored, cprint
from colorama import Fore, Back, Style
import copy
from csp_solver import GAC

"""
Class: ConstraintInstance
Contains the position of the cell involved in the constraint. The position is given as the row number and column number.
"""
class ConstraintInstance:
    # Position of the variables involved in the constraint
    rowNumber = None
    colNumber = None

    def __init__(self, rowVariableInstance, colVariableInstance):
        self.rowNumber = rowVariableInstance
        self.colNumber = colVariableInstance

    # Description: Returns all the variables involved in the constraint
    # Input: reference to a nonogram csp node
    # Output: allVariables
    def get_all_variables(self, problemObject):
        allVariables = [problemObject.rowVariables[self.rowNumber], problemObject.colVariables[self.colNumber]]
        return allVariables

    # Description: Returns all the non-focal variables involved in the constraint given a focal variable and a reference
    #               to a nonogram csp node. For this implementation the it's only one non-focal variable
    # Input: focalVariable, reference to a nonogram csp node problemObject
    # Output: all the non-focal variables, nonFocalVariables
    def get_non_focal_variables(self, focalVariable, problemObject):
        if (focalVariable.number == self.rowNumber and focalVariable.type == 'row'):
            nonFocalVariable = problemObject.colVariables[self.colNumber]
        elif (focalVariable.number == self.colNumber and focalVariable.type == 'col'):
            nonFocalVariable = problemObject.rowVariables[self.rowNumber]
        else:
            raise AssertionError
        return nonFocalVariable

    # Description: Evaluates a constraint to find if a given value in a domainValue satisfy the domainValue
    #               for the focalVariable
    # Input: - a specfic value in the domainValue, value
    #        - a possible value for the variable, domainValue
    #        - variable to evaluate, focalVariable
    # Output: allVariables
    def satisfied(self, value, domainValue, focalVariable):
        if(value == domainValue[focalVariable.number]):
            return True
        else:
            return False

    # Description: Returns all the domain values that can be used to possibly reduce another variables domain
    # Input: focalVariable, reference to the nonogram csp node problemObject
    # Output: Returns the domain values that can be used
    def get_pruning_values(self, focalVariable, problemObject):
        nonFocalVariable = self.get_non_focal_variables(focalVariable, problemObject)

        if all(domainValue[nonFocalVariable.number] == 0 for domainValue in focalVariable.domain):
            return [0]
        elif all(domainValue[nonFocalVariable.number] == 1  for domainValue in focalVariable.domain):
            return [1]
        else:
            return []

"""
Class: VariableInstance
A variable is a row/column in the nonogram puzzle
"""
class VariableInstance:
    type = None     # either 'row' or 'col'
    number = None   # the row or column number of the variable
    domain = None   # list of possible domain values for the variable

    def __init__(self, type, number, domain):
        self.type = type
        self.number = number
        self.domain = domain


"""
Class: NonogramCspNode
Represents the nonogram to be solved by the GAC algorithm.
"""
class NonogramCspNode:

    # Internal variables
    rows = None         # Number of rows in the nonogram
    cols = None         # Number of columns in the nonogram
    colors = None       # Dictionary of colors for printing the solution

    # CSP variables
    rowVariables = None # List row variable instances
    colVariables = None # List of column variable instances
    constraints = None  # List of constraint instances

    # A* variables
    state = None        # Unique state representing the current node in the A* search tree
    parent = None       # The parent of the current node
    kids = None         # The children of the current node
    g = 0
    f = 0
    h = 0

    def __init__(self):
        self.rows = None
        self.cols = None
        self.colors = {("0"): "white", ("1"): "red"}

        self.rowVariables = []
        self.colVariables = []
        self.constraints = []

        self.goal = None
        self.state = None
        self.parent = None
        self.kids = []
        self.g = 0
        self.f = 0
        self.h = 0


    """
    CSP specific functions:
    """

    # Description: Loads the configuration for the nonogram and builds the variables' domain.
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

                # Initialize storage and initial composition
                initialComposition = [0] * compositionRestrictionLength
                initialComposition[0] = maxSegmentMoves
                rowCompositions = [initialComposition]

                # Create a set storage to facilitate faster lookup when checking if a composition exists
                initialCompositionString = ','.join(map(str, initialComposition))
                rowCompositionsLookupTable = set()
                rowCompositionsLookupTable.add(initialCompositionString)

                # Find all compositions
                if (maxSegmentMoves != 0):
                    self.find_all_weak_compositions(rowCompositions, rowCompositionsLookupTable, initialComposition,
                                                    maxSegmentMoves, compositionRestrictionLength)

                # Translate from compositions to a row variables domain
                rowDomain = self.compositions_to_variable_values(rowCompositions, rowSpec)
                self.rowVariables.append(VariableInstance('row', rowNumber, rowDomain))

            for colNumber in range(0, self.cols):
                # Read column specifications from file
                colSpec = f.readline().split(" ")
                colSpec = [int(x) for x in colSpec]
                colSpec.reverse()  # to get the correct order according to the exercise text

                # Translate to a compositions-problem
                minTotalSegmentLength = sum(colSpec) + len(colSpec) - 1
                maxSegmentMoves = self.rows - minTotalSegmentLength
                compositionRestrictionLength = (len(colSpec) + 1)

                # Initialize storage and initial composition
                initialComposition = [0] * compositionRestrictionLength
                initialComposition[0] = maxSegmentMoves
                colCompositions = [initialComposition]

                # Create a set storage to facilitate faster lookup when checking if a composition exists
                initialCompositionString = ','.join(map(str, initialComposition))
                colCompositionsLookupTable = set()
                colCompositionsLookupTable.add(initialCompositionString)

                # Find all compositions
                if (maxSegmentMoves != 0):
                    self.find_all_weak_compositions(colCompositions, colCompositionsLookupTable, initialComposition,
                                                    maxSegmentMoves, compositionRestrictionLength)

                # Translate from compositions to a column variables domain
                colDomain = self.compositions_to_variable_values(colCompositions, colSpec)
                self.colVariables.append(VariableInstance('col', colNumber, colDomain))

    # Description: Displays the nonogram in the terminal window
    # Input: None
    # Output: None
    def display_node(self):
        rows = list(self.rowVariables)
        rows.reverse()
        for row in rows:
            for values in row.domain:
                for value in values:
                    if (value == 0):
                        cprint(colored('  ', self.colors[str(0)], 'on_white'), end ='')
                    elif(value == 1):
                        cprint(colored('  ', self.colors[str(1)], 'on_blue'), end='')
                print('\t', end='')
            print('')
        print('')

    # Description: Creates all necessary constraint instances for the nonogram and adds them to the list of constraints
    # Input: None
    # Output: None
    def create_all_constraints(self):
        for row in self.rowVariables:
            for col in self.colVariables:
                constraintInstance = ConstraintInstance(row.number, col.number)
                self.constraints.append(constraintInstance)


    # Description: Finds all possible (restricted) weak compositions of the number n with k parts.
    #             Continuously appends the compositions to the storage.
    # Input: storage compositionsAccumulator, parentNode parentComposition, number n, parts k
    # Output: 0
    def find_all_weak_compositions(self, compositionsAccumulator, compositionsLookup, parentComposition, n, k):
        newCompositions = []
        for i in range(1, k):
            tempComposition = list(parentComposition)
            tempComposition[0] -= 1
            tempComposition[i] += 1
            newCompositions.append(tempComposition)

        for composition in newCompositions:
            compositionString = ','.join(map(str, composition))
            if compositionString not in compositionsLookup:
                compositionsAccumulator.append(composition)
                compositionsLookup.add(compositionString)
                if(composition[0] != 0):
                    self.find_all_weak_compositions(compositionsAccumulator, compositionsLookup, composition, n, k)

    # Description: Translates the compositions found in find_all_weak_compositions to domain values.
    # Input: - list of possible compositions, compositions
    #        - segment sizes for the variable, variableSpec
    # Output: variable's domain, values
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



    """
    A* specific functions:
    """

    # Description: Determines whether the CSP solver has found a solution, by calculating the current total domain size.
    #              A solution is found when all domain sizes for all variables are reduced to one.
    # Input: None
    # Output: True if a solution is found, False otherwise
    def is_goal(self):
        totalDomainSize = self.get_total_domain_size()
        # Check if all domain sizes are reduced to one
        if totalDomainSize == (self.rows + self.cols):
            return True
        else:
            return False

    # Description: Generates successors for A* algorithm.
    #              Makes assumptions on ONE variable in the nonogram, and generates a child nonogram for each of the values
    #              that are assumed for the variable.
    #              Calls rerun (GAC algorithm) on each of the children. If the assumption proves to be valid, the child is added to the list of successors.
    # Input: None
    # Output: list of successors, successors
    def generate_successors(self):
        successors = []
        variables = self.rowVariables + self.colVariables

        # Find the variable with the smallest domain (larger than one)
        smallestDomainVar = min((variable for variable in variables if len(variable.domain) > 1),
                                key=lambda variable: len(variable.domain))

        # Reduce the variables domain to a singleton for every value and append it as a successor state
        for domainValue in smallestDomainVar.domain:
            # Create a copy to modify, only needs deepcopy of the variables
            nonogramChildNode = NonogramCspNode()
            nonogramChildNode.rowVariables = copy.deepcopy(self.rowVariables)
            nonogramChildNode.colVariables = copy.deepcopy(self.colVariables)
            nonogramChildNode.constraints = self.constraints
            nonogramChildNode.rows = self.rows
            nonogramChildNode.cols = self.cols

            # Create new csp solver for the child node
            cspSolver = GAC()
            cspSolver.set_problem_ref(nonogramChildNode)

            if (smallestDomainVar.type == 'col'):
                childSmallestDomainVar = nonogramChildNode.colVariables[smallestDomainVar.number]
                childSmallestDomainVar.domain = [domainValue]
            else:
                childSmallestDomainVar = nonogramChildNode.rowVariables[smallestDomainVar.number]
                childSmallestDomainVar.domain = [domainValue]

            validReduction = cspSolver.rerun(childSmallestDomainVar)

            # Append to successors only if rerun was successful
            if (validReduction):
                nonogramChildNode.state = nonogramChildNode.get_state_identifier()
                successors.append(nonogramChildNode)
        return successors

    # Description: Returns an unique state identifier for a nonogram node
    # Input: None
    # Output: String cspIdentifier
    def get_state_identifier(self):
        cspIdentifier = ''
        variables = self.rowVariables + self.colVariables
        for variable in variables:
            for vector in variable.domain:
                cspIdentifier += ''.join(map(str, vector))
        return cspIdentifier

    # Description: Finds the sum of all elements in each domain for every variable
    # Input: None
    # Output: total remaining values across all variables totalDomainSize
    def get_total_domain_size(self):
        totalDomainSize = 0
        for variable in self.rowVariables:
            totalDomainSize += len(variable.domain)
        for variable in self.colVariables:
            totalDomainSize += len(variable.domain)
        return totalDomainSize

    # Description: Evaluates a node
    # Input: None
    # Output: String cspIdentifier
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
