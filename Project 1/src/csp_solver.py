import sys
import numpy as np
from termcolor import colored, cprint
import colorama
from string import *

class Arc:
    focalVariable = None        # Reference to the variable
    constraint = None           # Reference to the constraint

    def __init__(self, focalVariable, constraint):
        self.focalVariable = focalVariable
        self.constraint = constraint


class GAC:
    queue = None
    problemRef = None

    def __init__(self):
        self.queue = []

    def initialization(self, problemObject):
        for constraint in problemObject.constraints:
            variables = constraint.get_all_variables(problemObject)
            for variable in variables:
                arc = Arc(focalVariable = variable, constraint = constraint)
                self.queue.append(arc)
        self.problemRef = problemObject

    def set_problem_ref(self, problemObject):
        self.problemRef = problemObject

    def domain_filtering_loop(self):
        while self.queue:
            arc = self.queue.pop()
            if self.revise(arc):
                if len(arc.focalVariable.domain) == 0:
                    return False
                self.add_all_neighboring_arcs(arc)
        return True

    def add_all_neighboring_arcs(self, arc):
        focalVariable, prevConstraint = arc.focalVariable, arc.constraint
        for constraint in self.problemRef.constraints:
            allVariables = constraint.get_all_variables(self.problemRef)
            if(focalVariable in allVariables):
                for variable in allVariables:
                    if(variable != focalVariable and constraint != prevConstraint):
                        arc = Arc(focalVariable = variable, constraint = constraint)
                        self.queue.append(arc)

    def revise(self, arc):
        revised = False
        focalVariable, constraint = arc.focalVariable, arc.constraint
        nonFocalVariable = constraint.get_non_focal_variable(focalVariable, self.problemRef)

        pruningValues = constraint.get_pruning_values(nonFocalVariable, self.problemRef)
        if(pruningValues):
            for value in pruningValues:
                focalVariableDomainCopy = list(focalVariable.domain)
                for domainValue in focalVariableDomainCopy:
                    satisfied = constraint.satisfied(value, domainValue, nonFocalVariable)
                    if not satisfied:
                        focalVariable.domain.remove(domainValue)
                        revised = True
        return revised

    def rerun(self, focalVariable):
        for constraint in self.problemRef.constraints:
            allVariables = constraint.get_all_variables(self.problemRef)
            if (focalVariable in allVariables):
                for variable in allVariables:
                    if (variable != focalVariable):
                        arc = Arc(focalVariable=variable, constraint=constraint)
                        self.queue.append(arc)
        validReduction = self.domain_filtering_loop()
        return validReduction