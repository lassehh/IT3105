import numpy as np
from sys import stdout
import math

class Map:
    start = None
    goal = None
    gridHeight = None
    gridLength = None
    map = None

    def __init__(self, pixels, start, goal):
        radius_y = 20
        radius_x = 15
        pixels[start[0]:start[0] + radius_y, start[1]] = 0
        pixels[start[0] - radius_y: start[0], start[1]] = 0
        pixels[start[0], start[1]:start[1] + radius_x] = 0
        pixels[start[0], start[1] - radius_x:start[1]] = 0
        self.start = start
        self.goal = goal
        self.gridHeight = pixels.shape[1]
        self.gridLength = pixels.shape[0]
        self.map = np.zeros(shape=(self.gridLength, self.gridHeight), dtype=np.object)
        for x in range(self.gridLength):
            for y in range(self.gridHeight):
                self.map[x, y] = Pixel(x,y, int(pixels[x,y]))

    def printMap(self):
        for y in range(0, self.gridHeight):
            for x in range(0, self.gridLength):
                stdout.write(str(self.map[x, y].value))
            print ('\n')

    def closenessCheck(self, c):
        #to prevent generating a path too close to the boundaries of the objects
        #i.e: check nodes in a distance around the robot
        raise NotImplementedError

class Pixel:
    x = None							# X coordinate
    y = None                            # Y coordinate
    state = None                        # Unique state identifier for the node/pixel
    parent = None                       # Best ascendant/parent found
    kids = None                         # Neighbours generated by this node
    g = None                            # Cost through this node
    f = None                            # Estimated cheapest solution through this node
    h = None                            # Heuristic for this node relative to the goal (see calc_h(self))
    value = None

    def __init__(self, x, y, val):
        self.x = x
        self.y = y
        self.state = str(x)+str(y)
        self.f = None
        self.h = None
        self.value = val
        self.parents = None
        self.kids = []
        if val == 0:
            self.g = 1.0
        elif val == 1:
            self.g = float('Inf')
        else:
            raise EnvironmentError

    def is_goal(self, map, radius = 5):
        return pow(math.fabs(self.x - map.goal[0]), 2) + pow(math.fabs(self.y - map.goal[1]), 2) <= pow(radius,2)

    def calc_h(self, map):
        return math.fabs(self.x - map.goal[0]) + math.fabs(self.y - map.goal[1])

    def calc_f(self, map):							#calculate f = g+h
        return self.calc_h(map) + self.g

    def generate_close_successors(self, map):
        successors = []
        if (self.y + 1 <= map.gridHeight - 1):
            pixel_up = map.map[(self.x),(self.y + 1)]
            successors.append(pixel_up)
        if (self.y - 1 >= 0):
            pixel_down = map.map[(self.x),(self.y - 1)]
            successors.append(pixel_down)
        if (self.x + 1 <= map.gridLength - 1):
            pixel_right = map.map[(self.x + 1), (self.y)]
            successors.append(pixel_right)
        if (self.x - 1 >= 0):
            pixel_left = map.map[(self.x - 1), (self.y)]
            successors.append(pixel_left)
        return successors

    def generate_stride_successors(self, map, stride = 5):
        successors = []
        if (self.y + stride <= map.gridHeight - 1):
            pixel_up = map.map[(self.x), (self.y + stride)]
            successors.append(pixel_up)
        if (self.y - stride >= 0):
            pixel_down = map.map[(self.x), (self.y - stride)]
            successors.append(pixel_down)
        if (self.x + stride <= map.gridLength - 1):
            pixel_right = map.map[(self.x + stride), (self.y)]
            successors.append(pixel_right)
        if (self.x - stride >= 0):
            pixel_left = map.map[(self.x - stride), (self.y)]
            successors.append(pixel_left)
        return successors


    def arc_cost(self, pixel):
        return pixel.g

