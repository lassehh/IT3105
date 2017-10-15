class GacAstar:
    cspNode = None
    astarSearch = None

    def __init__(self, cspNode, astarSearch):
        self.cspNode = cspNode
        self.astarSearch = astarSearch

    def run(self, validReduction):
        if validReduction and not self.cspNode.is_goal():
            solution, numberOfMovesToSolution, searchNodesGenerated, searchNodesExpanded = self.astarSearch.best_first_search()
            return solution, numberOfMovesToSolution, searchNodesGenerated, searchNodesExpanded
        else:
            return self.cspNode, -1, -1, -1
