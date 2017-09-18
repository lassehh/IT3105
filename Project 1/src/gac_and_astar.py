from nonogram_csp import *
from csp_solver import *
from astar_search import *


class GacAstar:
    cspNode = None
    cspSolver = None
    astarSearch = None

    def __init__(self, cspNode, cspSolver, astarSearch):
        self.cspNode = cspNode
        self.cspSolver = cspSolver
        self.astarSearch = astarSearch

    def run(self, validReduction):
        if validReduction and not self.cspNode.is_goal():
            solution, numberOfMovesToSolution, searchNodesGenerated, searchNodesExpanded = self.astarSearch.best_first_search()
            return solution, numberOfMovesToSolution, searchNodesGenerated, searchNodesExpanded
        else:
            return self.cspNode, -1, -1, -1
