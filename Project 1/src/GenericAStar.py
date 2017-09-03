class AStar:
	created_dict = None  			# Dictionary containing all nodes created
	open_list = []  				# Nodes in the queue not yet visited
	closed_list = []  				# Visited nodes
	n0 = None
	start = None
	search_type = None

	def __init__(self, search_type, start_search_node):
		self.start = start_search_node
		self.search_type = search_type
		self.created_dict = {(self.start.state): self.start}
		self.open_list = []
		self.closed_list = []

	def best_first_search(self):
		n0 = self.start
		n0.g = 0
		n0.calc_h()
		n0.f = n0.h + n0.g

		self.open_list.append(n0)

		# Agenda Loop
		while (self.open_list):
			x = self.search_queue_pop(self.search_type, self.open_list)
			self.closed_list.append(x)
			if x.is_goal():
				return x
			successors = x.generate_successors()

			for s in successors:
				if (s.state) in self.created_dict:
					s = self.created_dict[(s.state)]
				if s not in self.open_list and s not in self.closed_list:
					self.attach_and_eval(s, x)
					self.open_list.append(s)
					self.created_dict[(s.state)] = s
				elif x.g + x.arc_cost() < s.g:
					self.attach_and_eval(s, x)
					if s in self.closed_list:
						self.propagate_path_improvements(s)
		return []

	def search_queue_pop(self, search_type, queue):  #
		if search_type == "BFS":  # Breadth First Search
			return queue.pop(0)
		elif search_type == "DFS":  # Depth First Search
			return queue.pop()
		elif search_type == "BestFS":  # Best First Search
			current_node = min(queue, key=lambda x: x.f)
			queue.remove(current_node)
			return current_node
		else:
			raise NotImplementedError

	def attach_and_eval(self, c, p):
		c.parent = p
		c.g = p.g + p.arc_cost()
		c.h = c.calc_h()
		c.f = c.g + c.h

	def propagate_path_improvements(self, p):
		for c in p.kids:
			if p.g + p.arc_cost() < c.g:
				c.parent = p
				c.g = p.g + p.arc_cost()
				c.f = c.g + c.h
				self.propagate_path_improvements(c)

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