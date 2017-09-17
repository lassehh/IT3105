import sys
import numpy as np
from termcolor import colored, cprint
import colorama
from string import *

class Arc:
    focalVariable = None
    constraint = None

    def __init__(self, focalVariable, constraint):
        self.focalVariable = focalVariable
        self.constraint = constraint


class GAC:
    queue = None
    cspConstraints = None

    def __init__(self):
        self.queue = []

    def initialization(self, cspConstraints):
        for constraint in cspConstraints:
            variables = constraint.get_all_variables()
            for variable in variables:
                arc = Arc(focalVariable = variable, constraint = constraint)
                self.queue.append(arc)
        self.cspConstraints = cspConstraints

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
        for constraint in self.cspConstraints:
            allVariables = constraint.get_all_variables()
            if(focalVariable in allVariables):
                for variable in allVariables:
                    if(variable != focalVariable and constraint != prevConstraint):
                        arc = Arc(focalVariable = variable, constraint = constraint)
                        self.queue.append(arc)

    def revise(self, arc):
        revised = False
        focalVariable, constraint = arc.focalVariable, arc.constraint
        nonFocalVariable = constraint.get_non_focal_variable(focalVariable)

        pruningValues = constraint.get_pruning_values(focalVariable)
        if(pruningValues):
            for value in pruningValues:
                nonFocalVariableDomainCopy = list(nonFocalVariable.domain)
                for domainValue in nonFocalVariableDomainCopy:
                    satisfied = constraint.satisfied(value, domainValue, focalVariable)
                    if not satisfied:
                        nonFocalVariable.domain.remove(domainValue)
                        revised = True
        return revised

    def rerun(self, focalVariable):
        for constraint in self.cspConstraints:
            allVariables = constraint.get_all_variables()
            if (focalVariable in allVariables):
                for variable in allVariables:
                    if (variable != focalVariable):
                        arc = Arc(focalVariable=variable, constraint=constraint)
                        self.queue.append(arc)
        validReduction = self.domain_filtering_loop()
        return validReduction