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
            solution, __, __ = self.astarSearch.best_first_search()
            return solution
        else:
            return self.cspNode
