class AStar:
	createdDict = None  			# Dictionary containing all nodes created
	openList = []  				    # Nodes in the queue not yet visited
	closedList = []  				# Visited nodes
	n0 = None
	start = None
	searchType = None
	displayMode = None

	searchNodesGenerated = None
	costFromStartToGoal = None

	def __init__(self, searchType, startSearchNode, displayMode = False):
		self.start = startSearchNode
		self.searchType = searchType
		self.displayMode = displayMode
		self.createdDict = {(self.start.state): self.start}
		self.openList = []
		self.closedList = []
		self.searchNodesGenerated = 0
		self.costFromStartToGoal = 0

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
					x.display_game_board()
				else:
					self.displayMode = False

			# Goal check
			if x.is_goal():
				costFromStartToGoal = self.get_number_of_moves(x)
				self.displaySolutionPath(x, costFromStartToGoal)
				return x, costFromStartToGoal, self.searchNodesGenerated

			successors = x.generate_successors()
			for s in successors:
				x.kids.append(s)
				if (s.state) in self.createdDict:
					s = self.createdDict[(s.state)]
				if s not in self.openList and s not in self.closedList:
					self.attach_and_eval(s, x)
					self.openList.append(s)
					self.createdDict[(s.state)] = s
					self.searchNodesGenerated += 1
				elif x.g + x.arc_cost() < s.g:
					self.attach_and_eval(s, x)
					if s in self.closedList:
						self.propagate_path_improvements(s)
		return [], -1, self.searchNodesGenerated

	def search_queue_pop(self, searchType, queue):  #
		if searchType == "BFS":  # Breadth First Search
			return queue.pop(0)
		elif searchType == "DFS":  # Depth First Search
			return queue.pop()
		elif searchType == "BestFS":  # Best First Search
			current_node = min(queue, key=lambda x: x.f)
			queue.remove(current_node)
			return current_node
		else:
			raise NotImplementedError

	def attach_and_eval(self, c, p):
		c.parent = p
		c.g = p.g + p.arc_cost()
		c.calc_h()
		c.f = c.g + c.h

	def propagate_path_improvements(self, p):
		for c in p.kids:
			if p.g + p.arc_cost() < c.g:
				c.parent = p
				c.g = p.g + p.arc_cost()
				c.f = c.g + c.h
				self.propagate_path_improvements(c)

	def get_number_of_moves(self, solutionNode):
		numberOfMoves = 0
		while(solutionNode.parent != None):
			numberOfMoves += 1
			solutionNode = solutionNode.parent
		return numberOfMoves

	def displaySolutionPath(self, solutionNode, movesToSolution):
		userInputDisplaySolutionPath = input("[DISPLAY_MODE]: Display the solution path? (y/n) ")
		stopDisplaying = False
		move = movesToSolution
		if (userInputDisplaySolutionPath == 'y'):
			while (stopDisplaying != True):
				print('[DISPLAY_MODE]: Move/board state: ' + str(move))
				move -= 1
				solutionNode.display_game_board()
				if (solutionNode.parent == None): stopDisplaying = True
				solutionNode = solutionNode.parent



