import numpy as np
import matplotlib.pyplot as plt
import time


import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/")
from Search_3D.env3D import env
from Search_3D import Astar3D
from Search_3D.utils3D import StateSpace, getDist, getNearest, getRay, isinbound, isinball, isCollide, children, cost, \
    initcost
from Search_3D.plot_util3D import visualization


class D_star(object):
    def __init__(self, resolution=1):
        self.Alldirec = {(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1, \
                        (-1, 0, 0): 1, (0, -1, 0): 1, (0, 0, -1): 1, \
                        (1, 1, 0): np.sqrt(2), (1, 0, 1): np.sqrt(2), (0, 1, 1): np.sqrt(2), \
                        (-1, -1, 0): np.sqrt(2), (-1, 0, -1): np.sqrt(2), (0, -1, -1): np.sqrt(2), \
                        (1, -1, 0): np.sqrt(2), (-1, 1, 0): np.sqrt(2), (1, 0, -1): np.sqrt(2), \
                        (-1, 0, 1): np.sqrt(2), (0, 1, -1): np.sqrt(2), (0, -1, 1): np.sqrt(2), \
                        (1, 1, 1): np.sqrt(3), (-1, -1, -1) : np.sqrt(3), \
                        (1, -1, -1): np.sqrt(3), (-1, 1, -1): np.sqrt(3), (-1, -1, 1): np.sqrt(3), \
                        (1, 1, -1): np.sqrt(3), (1, -1, 1): np.sqrt(3), (-1, 1, 1): np.sqrt(3)}
        self.settings = 'CollisionChecking'
        self.env = env(resolution=resolution)
        self.X = StateSpace(self.env)
        self.x0, self.xt = getNearest(self.X, self.env.start), getNearest(self.X, self.env.goal)
        self.b = defaultdict(lambda: defaultdict(dict))  
        self.OPEN = {} 
        self.h = {} 
        self.tag = {} 
        self.V = set() 
        self.ind = 0
        self.Path = []
        self.done = False
        self.Obstaclemap = {}

    def checkState(self, y):
        if y not in self.h:
            self.h[y] = 0
        if y not in self.tag:
            self.tag[y] = 'New'

    def get_kmin(self):
        if self.OPEN:
            return min(self.OPEN.values())
        return -1

    def min_state(self):
        if self.OPEN:
            minvalue = min(self.OPEN.values())
            for k in self.OPEN.keys():
                if self.OPEN[k] == minvalue:
                    return k, self.OPEN.pop(k)
        return None, -1

    def insert(self, x, h_new):
        if self.tag[x] == 'New':
            kx = h_new
        if self.tag[x] == 'Open':
            kx = min(self.OPEN[x], h_new)
        if self.tag[x] == 'Closed':
            kx = min(self.h[x], h_new)
        self.OPEN[x] = kx
        self.h[x], self.tag[x] = h_new, 'Open'

    def process_state(self):
        x, kold = self.min_state()
        self.tag[x] = 'Closed'
        self.V.add(x)
        if x is None:
            return -1
        # check if 1st timer s
        self.checkState(x)
        if kold < self.h[x]:
            for y in children(self, x):
                # check y
                self.checkState(y)
                a = self.h[y] + cost(self, y, x)
                if self.h[y] <= kold and self.h[x] > a:
                    self.b[x], self.h[x] = y, a
        if kold == self.h[x]: 
            for y in children(self, x):
                # check y
                self.checkState(y)
                bb = self.h[x] + cost(self, x, y)
                if self.tag[y] == 'New' or \
                        (self.b[y] == x and self.h[y] != bb) or \
                        (self.b[y] != x and self.h[y] > bb):
                    self.b[y] = x
                    self.insert(y, bb)
        else:
            for y in children(self, x):
                self.checkState(y)
                bb = self.h[x] + cost(self, x, y)
                if self.tag[y] == 'New' or \
                        (self.b[y] == x and self.h[y] != bb):
                    self.b[y] = x
                    self.insert(y, bb)
                else:
                    if self.b[y] != x and self.h[y] > bb:
                        self.insert(x, self.h[x])
                    else:
                        if self.b[y] != x and self.h[y] > bb and \
                                self.tag[y] == 'Closed' and self.h[y] == kold:
                            self.insert(y, self.h[y])
        return self.get_kmin()

    def modify_cost(self, x):
        xparent = self.b[x]
        if self.tag[x] == 'Closed':
            self.insert(x, self.h[xparent] + cost(self, x, xparent))

    def modify(self, x):
        self.modify_cost(x)
        while True:
            kmin = self.process_state()
            if kmin >= self.h[x]:
                break

    def path(self, goal=None):
        path = []
        if not goal:
            x = self.x0
        else:
            x = goal
        start = self.xt
        while x != start:
            path.append([np.array(x), np.array(self.b[x])])
            x = self.b[x]
        return path

    def calculate_path_length(self):
        length = 0
        if len(self.Path) > 1:
            for segment in self.Path:
                if len(segment) > 1:
                    length += getDist(segment[0], segment[1])
        return length

    def calculate_path_smoothness(self):
        angles = []
        if len(self.Path) < 3:
            return 0 

        for i in range(1, len(self.Path) - 1):
            a = np.array(self.Path[i-1][1]) 
            b = np.array(self.Path[i][0])   
            c = np.array(self.Path[i][1])

            ba = a - b
            bc = c - b

            if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
                continue 

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1, 1))

            angles.append(np.degrees(angle))

        return np.mean(angles) if angles else 0

    def calculate_path_safety(self):
        min_distances = []
        for segment in self.Path:
            point = np.array(segment[0]) 
            for sphere in self.env.balls:
                center = np.array(sphere[:3])
                radius = sphere[3]
                distance = np.linalg.norm(point - center) - radius
                min_distances.append(distance)
            for block in self.env.blocks:
                block_center = np.array([(block[0] + block[3])/2, (block[1] + block[4])/2, (block[2] + block[5])/2])
                distance = np.linalg.norm(point - block_center)
                min_distances.append(distance)
        return min(min_distances) if min_distances else float('inf')

    def calculate_path_directness(self):
        if not self.Path:
            return float('inf')
        start = np.array(self.env.start)
        goal = np.array(self.env.goal)
        straight_line_distance = np.linalg.norm(goal - start)
        path_length = self.calculate_path_length()
        return path_length / straight_line_distance if straight_line_distance > 0 else float('inf')



    


    def run(self):

        start_time = time.time()
        self.OPEN[self.xt] = 0
        self.tag[self.x0] = 'New'
        # first run
        while True:
            self.process_state()
            # visualization(self)
            if self.tag[self.x0] == "Closed":
                break
            self.ind += 1
        self.Path = self.path()
        self.done = True
        visualization(self)
        plt.pause(0.2)
        for i in range(5):
            self.env.move_block(a=[0, -0.50, 0], s=0.5, block_to_move=1, mode='translation')
            self.env.move_block(a=[-0.25, 0, 0], s=0.5, block_to_move=0, mode='translation')
            s = tuple(self.env.start)
            while s != self.xt:
                if s == tuple(self.env.start):
                    sparent = self.b[self.x0]
                else:
                    sparent = self.b[s]
                if cost(self, s, sparent) == np.inf:
                    self.modify(s)
                    continue
                self.ind += 1
                s = sparent
            self.Path = self.path()
            visualization(self)
        plt.show()
        self.execution_time = time.time() - start_time 
        self.path_length = self.calculate_path_length() 
        self.path_smoothness = self.calculate_path_smoothness()
        self.path_safety = self.calculate_path_safety()
        self.path_directness = self.calculate_path_directness()
        print(f"Execution Time: {self.execution_time} seconds")
        print(f"Path Length: {self.path_length} units")
        print(f"Path Smoothness: {self.path_smoothness}")
        print(f"Path Safety: {self.path_safety}")
        print(f"Path Directness: {self.path_directness}")


if __name__ == '__main__':
    D = D_star(1)
    D.run()
