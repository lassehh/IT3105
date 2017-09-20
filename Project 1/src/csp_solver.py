
"""
Class: Arc
An arc represents variables and the constraint between them.
"""
class Arc:
    focalVariable = None        # Reference to the focal variable
    constraint = None           # Reference to the constraint

    def __init__(self, focalVariable, constraint):
        self.focalVariable = focalVariable
        self.constraint = constraint

"""
Class: GAC
Impelements the General Arc Consistency algorithm.
"""
class GAC:
    queue = None        # list of arcs that must be revised
    problemRef = None   # Reference to the problem that GAC tries to solve

    def __init__(self):
        self.queue = []


# Description: appends all arcs to the queue
# Input: Reference to the problem that GAC tries to solve, problemObject
# Output: None
    def initialization(self, problemObject):
        for constraint in problemObject.constraints:
            variables = constraint.get_all_variables(problemObject)
            for variable in variables:
                arc = Arc(focalVariable = variable, constraint = constraint)
                self.queue.append(arc)
        self.problemRef = problemObject

    def set_problem_ref(self, problemObject):
        self.problemRef = problemObject

# Description: Implementation of the domain filtering loop of the GAC algorithm.
#              Pops arcs from the queue and calls revise. Adds new arcs to the queue if necessary
# Input: None
# Output: False if an inconsistency is found and True otherwise, validReduction
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

# Description: removes domain values from a variable if the constraint is not satisfied
# Input: the focal variable and constraint in the revision, arc
# Output: True/False variable, revised
    def revise(self, arc):
        revised = False
        focalVariable, constraint = arc.focalVariable, arc.constraint
        nonFocalVariable = constraint.get_non_focal_variables(focalVariable, self.problemRef)

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

# Description: Appends new arcs to the queue, given that an assumption has just been made on a variable. Runs domain filtering loop
# Input: the variable where an assumption has been made, focalVariable
# Output: True/False value from domain filtering loop
    def rerun(self, focalVariable):
        for constraint in self.problemRef.constraints:
            allVariables = constraint.get_all_variables(self.problemRef)
            if (focalVariable in allVariables):
                for variable in allVariables:
                    if (variable != focalVariable):
                        arc = Arc(focalVariable = variable, constraint = constraint)
                        self.queue.append(arc)
        validReduction = self.domain_filtering_loop()
        return validReduction
