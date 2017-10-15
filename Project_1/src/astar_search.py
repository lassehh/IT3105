
"""
Class: AStar
Implements a general A* algorithm
"""
class AStar:
	createdDict = None  			# Dictionary containing all nodes created
	openList = []  				    # Nodes in the queue not yet visited
	closedList = []  				# Visited nodes
	n0 = None						# Initial node
	start = None					# Start node
	searchType = None				# How to pop elements from the queue
	displayMode = None				# Controls the displaying mode of the algorithm

	searchNodesGenerated = None
	searchNodesExpanded = None
	costFromStartToGoal = None

	def __init__(self, searchType, startSearchNode, displayMode = False):
		self.start = startSearchNode
		self.searchType = searchType
		self.displayMode = displayMode
		self.createdDict = {(self.start.state): self.start}
		self.openList = []
		self.closedList = []
		self.searchNodesGenerated = 0
		self.searchNodesExpanded = 0
		self.costFromStartToGoal = 0

	# Description: The main loop of the algorithm
	# Input: None
	# Output: solution node x, costFromStartToGoal, searchNodesGenerated, searchNodesExpanded
	def best_first_search(self):
		self.n0 = self.start
		self.n0.g = 0
		self.n0.calc_h()
		self.n0.f = self.n0.h + self.n0.g

		self.openList.append(self.n0)
		self.searchNodesGenerated = 1

		# Agenda Loop
		while (self.openList):
			x = self.search_queue_pop(self.searchType, self.openList)
			self.closedList.append(x)
			# Display mode
			if (self.displayMode == True):
				userInputDisplayOption = input("[DISPLAY_MODE]: Continue with display mode? (y/n) ")
				if (userInputDisplayOption == "y"):
					self.displayMode = True
					print('[DISPLAY_MODE]: Current board state: ')
					x.display_node()
				else:
					self.displayMode = False

			# Goal check
			if x.is_goal():
				costFromStartToGoal = self.get_number_of_moves(x)
				self.displaySolutionPath(x, costFromStartToGoal)
				return x, costFromStartToGoal, self.searchNodesGenerated, self.searchNodesExpanded

			successors = x.generate_successors()
			self.searchNodesExpanded += 1
			for s in successors:
				x.kids.append(s)
				if (s.state) in self.createdDict:
					s = self.createdDict[(s.state)]
				if s not in self.openList and s not in self.closedList:
					self.attach_and_eval(s, x)
					self.openList.append(s)
					self.createdDict[(s.state)] = s
					self.searchNodesGenerated += 1
				elif x.g + x.arc_cost(s) < s.g:
					self.attach_and_eval(s, x)
					if s in self.closedList:
						self.propagate_path_improvements(s)
		return [], -1, self.searchNodesGenerated, self.searchNodesExpanded

	# Description: Pops the queue based on which search type is used
	# Input: searchType, queue
	# Output: the node poped, currentNode
	def search_queue_pop(self, searchType, queue):  #
		if searchType == "BFS":  # Breadth First Search
			return queue.pop(0)
		elif searchType == "DFS":  # Depth First Search
			return queue.pop()
		elif searchType == "BestFS":  # Best First Search
			currentNode = min(queue, key=lambda x: x.f)
			queue.remove(currentNode)
			return currentNode
		else:
			raise NotImplementedError

	# Description: Evaluates a child node
	# Input: child c, parent p,
	# Output: None
	def attach_and_eval(self, c, p):
		c.parent = p
		c.g = p.g + p.arc_cost(c)
		c.calc_h()
		c.f = c.g + c.h

	# Description: Recursively investigates if there exists a better path between parent and children
	# Input: parent p
	# Output: None
	def propagate_path_improvements(self, p):
		for c in p.kids:
			if p.g + p.arc_cost(c) < c.g:
				c.parent = p
				c.g = p.g + p.arc_cost(c)
				c.f = c.g + c.h
				self.propagate_path_improvements(c)

	# Description: Finds the total numbers steps from the start to the solutiion
	# Input: solutionnode
	# Output: numberOfMoves
	def get_number_of_moves(self, solutionNode):
		numberOfMoves = 0
		while(solutionNode.parent != None):
			numberOfMoves += 1
			solutionNode = solutionNode.parent
		return numberOfMoves

	# Description: Display the path taken by the algorithm from start to solution
	# Input: solutionNode, stepsToSolution
	# Output: None
	def displaySolutionPath(self, solutionNode, stepsToSolution):
		userInputDisplaySolutionPath = input("[DISPLAY_MODE]: Display the solution path? (y/n) ")
		stopDisplaying = False
		step = stepsToSolution
		if (userInputDisplaySolutionPath == 'y'):
			while (stopDisplaying != True):
				print('[DISPLAY_MODE]: Move/board state: ' + str(step))
				step -= 1
				solutionNode.display_node()
				if (solutionNode.parent == None): stopDisplaying = True
				solutionNode = solutionNode.parent
