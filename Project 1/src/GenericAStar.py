class AStar:
	created_dict = None  			# Dictionary containing all nodes created
	open_list = []  				# Nodes in the queue not yet visited
	closed_list = []  				# Visited nodes
	n0 = None
	start = None
	search_type = None
	map = None
	stride = None

	def __init__(self, map, search_type, stride):  # Initialise A*
		self.start = map.map[map.start[0], map.start[1]]  # Sets the start state
		self.search_type = search_type  # Sets search type (for module 1)
		self.created_dict = {(self.start.state): self.start}  # place the startnode in the created dictionary
		self.open_list = []  # open list is empty
		self.closed_list = []  # closed list is empty
		self.map = map
		self.stride = stride

	def best_first_search(self):
		n0 = self.start

		n0.calc_h(self.map)
		n0.calc_f(self.map)
		n0.g = 0

		self.open_list.append(n0)

		# Agenda Loop
		# while no solution
		while (self.open_list):
			x = self.search_queue_pop(self.search_type, self.open_list)
			self.closed_list.append(x)
			if x.is_goal(self.map):
				return self.waypoints(x)
			successors = x.generate_stride_successors(self.map, self.stride)

			for s in successors:
				if (s.state) in self.created_dict:
					s = self.created_dict[(s.state)]
				if s not in self.open_list and s not in self.closed_list:
					self.attach_and_eval(s, x, self.map)
					self.open_list.append(s)
					self.created_dict[(s.state)] = s
				elif x.g + x.arc_cost(s) < s.g:
					self.attach_and_eval(s, x)
					if s in self.closed_list:
						self.propagate_path_improvements(s)

		return []

	def search_queue_pop(self, search_type, queue):  # for popping elements off the agenda queue (open list)
		if search_type == "BFS":  # Breadth First Search mode
			return queue.pop(0)
		elif search_type == "DFS":  # Depth First Search mode
			return queue.pop()
		elif search_type == "BestFS":  # Best First Search mode
			current_node = min(queue, key=lambda x: x.f)
			queue.remove(current_node)
			return current_node
		else:  # If not recognized search mode
			raise NotImplementedError

	def attach_and_eval(self, c, p, map):  # Sett parent, g, h and f values of a new node
		c.parent = p
		c.g = p.g + p.arc_cost(c)
		c.h = c.calc_h(map)
		c.f = c.g + c.h

	def propagate_path_improvements(self, p):  # Updates parent, g, h and f values if a better parent is found
		for c in p.kids:
			if p.g + p.arc_cost(c) < c.g:
				c.parent = p
				c.g = p.g + p.arc_cost(c)
				c.f = c.g + c.h
				self.propagate_path_improvements(c)

	def path(self, x):  # Returns the path from the start node/state to the goal node/state
		goal_path = [x.state]
		while x.parent:
			x = x.parent
			print (x.x, x.y)
			goal_path.append(x.state)
		return goal_path[::-1]

	def waypoints(self, x): # Returns the path found by A*
		waypoints = [(x.x,x.y)]
		dir = self.direction(x, x.parent)
		while x.parent:
			tmp_dir = self.direction(x, x.parent)
			if dir != tmp_dir:
				waypoints.append((x.x,x.y))
			if x.parent:
				x = x.parent
			dir = tmp_dir
		waypoints.append((x.x,x.y))
		#print waypoints[::-1]
		return waypoints[::-1]

	def direction(self, p1, p2):
		if p1.x - p2.x > 0:
			return 1
		elif p1.x - p2.x < 0:
			return 2
		elif p1.y - p2.y > 0:
			return 3
		elif p1.y - p2.y < 0:
			return 4
