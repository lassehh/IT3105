class AStar:
	created_dict = None  			# Dictionary containing all nodes created
	open_list = []  				# Nodes in the queue not yet visited
	closed_list = []  				# Visited nodes
	n0 = None
	start = None
	searchType = None

	searchNodesGenerated = 0

	def __init__(self, search_type, start_search_node):
		self.start = start_search_node
		self.searchType = search_type
		self.created_dict = {(self.start.state): self.start}
		self.open_list = []
		self.closed_list = []
		self.searchNodesGenerated = 0

	def best_first_search(self):
		self.n0 = self.start
		self.n0.g = 0
		self.n0.calc_h()
		self.n0.f = self.n0.h + self.n0.g

		self.open_list.append(self.n0)
		self.searchNodesGenerated = 1

		# Agenda Loop
		while (self.open_list):
			x = self.search_queue_pop(self.searchType, self.open_list)
			#x.display_game_board()
			self.closed_list.append(x)
			if x.is_goal():
				return x
			successors = x.generate_successors()

			for s in successors:
				#s.display_game_board()
				x.kids.append(s)
				if (s.state) in self.created_dict:
					s = self.created_dict[(s.state)]
				if s not in self.open_list and s not in self.closed_list:
					self.attach_and_eval(s, x)
					self.open_list.append(s)
					self.created_dict[(s.state)] = s
					self.searchNodesGenerated += 1
				elif x.g + x.arc_cost() < s.g:
					self.attach_and_eval(s, x)
					if s in self.closed_list:
						self.propagate_path_improvements(s)
		return []

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


	# def path(self, x):
	# 	goal_path = [x.state]
	# 	while x.parent:
	# 		x = x.parent
	# 		print (x.x, x.y)
	# 		goal_path.append(x.state)
	# 	return goal_path[::-1]
    #
	# def waypoints(self, x):
	# 	waypoints = [(x.x,x.y)]
	# 	dir = self.direction(x, x.parent)
	# 	while x.parent:
	# 		tmp_dir = self.direction(x, x.parent)
	# 		if dir != tmp_dir:
	# 			waypoints.append((x.x,x.y))
	# 		if x.parent:
	# 			x = x.parent
	# 		dir = tmp_dir
	# 	waypoints.append((x.x,x.y))
	# 	return waypoints[::-1]
    #
	# def direction(self, p1, p2):
	# 	if p1.x - p2.x > 0:
	# 		return 1
	# 	elif p1.x - p2.x < 0:
	# 		return 2
	# 	elif p1.y - p2.y > 0:
	# 		return 3
	# 	elif p1.y - p2.y < 0:
	# 		return 4