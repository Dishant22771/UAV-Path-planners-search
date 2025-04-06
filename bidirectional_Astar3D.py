
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/")
from Search_3D.env3D import env
from Search_3D.utils3D import getDist, getRay, g_Space, Heuristic, getNearest, isCollide, cost, children, heuristic_fun
from Search_3D.plot_util3D import visualization
import queue


class Weighted_A_star(object):
    def __init__(self,resolution=0.5):
        self.Alldirec = {(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1, \
                        (-1, 0, 0): 1, (0, -1, 0): 1, (0, 0, -1): 1, \
                        (1, 1, 0): np.sqrt(2), (1, 0, 1): np.sqrt(2), (0, 1, 1): np.sqrt(2), \
                        (-1, -1, 0): np.sqrt(2), (-1, 0, -1): np.sqrt(2), (0, -1, -1): np.sqrt(2), \
                        (1, -1, 0): np.sqrt(2), (-1, 1, 0): np.sqrt(2), (1, 0, -1): np.sqrt(2), \
                        (-1, 0, 1): np.sqrt(2), (0, 1, -1): np.sqrt(2), (0, -1, 1): np.sqrt(2), \
                        (1, 1, 1): np.sqrt(3), (-1, -1, -1) : np.sqrt(3), \
                        (1, -1, -1): np.sqrt(3), (-1, 1, -1): np.sqrt(3), (-1, -1, 1): np.sqrt(3), \
                        (1, 1, -1): np.sqrt(3), (1, -1, 1): np.sqrt(3), (-1, 1, 1): np.sqrt(3)}
        self.env = env(resolution = resolution)
        self.settings = 'NonCollisionChecking'
        self.start, self.goal = tuple(self.env.start), tuple(self.env.goal)
        self.g = {self.start:0,self.goal:0}
        self.OPEN1 = queue.MinheapPQ()
        self.OPEN2 = queue.MinheapPQ()
        self.Parent1, self.Parent2 = {}, {}
        self.CLOSED1, self.CLOSED2 = set(), set()
        self.V = []
        self.done = False
        self.Path = []

    def run(self):

        start_time = time.time()
        x0, xt = self.start, self.goal
        self.OPEN1.put(x0, self.g[x0] + heuristic_fun(self,x0,xt)) 
        self.OPEN2.put(xt, self.g[xt] + heuristic_fun(self,xt,x0)) 
        self.ind = 0
        while not self.CLOSED1.intersection(self.CLOSED2):
            xi1, xi2 = self.OPEN1.get(), self.OPEN2.get() 
            self.CLOSED1.add(xi1) 
            self.CLOSED2.add(xi2)
            self.V.append(xi1)
            self.V.append(xi2)
            allchild1,  allchild2 = children(self,xi1), children(self,xi2)
            self.evaluation(allchild1,xi1,conf=1)
            self.evaluation(allchild2,xi2,conf=2)
            if self.ind % 100 == 0: print('iteration number = '+ str(self.ind))
            self.ind += 1
        self.common = self.CLOSED1.intersection(self.CLOSED2)
        self.done = True
        self.Path = self.path()
        visualization(self)
        plt.show()

        self.path_smoothness = self.calculate_path_smoothness()
        self.path_safety = self.calculate_path_safety()
        self.path_directness = self.calculate_path_directness()
        self.execution_time = time.time() - start_time 
        self.path_length = self.calculate_path_length()  
        print(f"Execution Time: {self.execution_time} seconds")
        print(f"Path Length: {self.path_length} units")
        print(f"Path Smoothness: {self.path_smoothness}")
        print(f"Path Safety: {self.path_safety}")
        print(f"Path Directness: {self.path_directness}")
        


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
        if len(self.Path) == 0:
            return float('inf')
        start = np.array(self.env.start)
        goal = np.array(self.env.goal)
        straight_line_distance = np.linalg.norm(goal - start)
        path_length = self.calculate_path_length()
        return path_length / straight_line_distance if straight_line_distance > 0 else float('inf')
    def evaluation(self, allchild, xi, conf):
        for xj in allchild:
            if conf == 1:
                if xj not in self.CLOSED1:
                    if xj not in self.g:
                        self.g[xj] = np.inf
                    else:
                        pass
                    gi = self.g[xi]
                    a = gi + cost(self,xi,xj)
                    if a < self.g[xj]:
                        self.g[xj] = a
                        self.Parent1[xj] = xi
                        self.OPEN1.put(xj, a+1*heuristic_fun(self,xj,self.goal))
            if conf == 2:
                if xj not in self.CLOSED2:
                    if xj not in self.g:
                        self.g[xj] = np.inf
                    else:
                        pass
                    gi = self.g[xi]
                    a = gi + cost(self,xi,xj)
                    if a < self.g[xj]:
                        self.g[xj] = a
                        self.Parent2[xj] = xi
                        self.OPEN2.put(xj, a+1*heuristic_fun(self,xj,self.start))
            
    def path(self):
        path = []
        goal = self.goal
        start = self.start
        x = list(self.common)[0]
        while x != start:
            path.append([x,self.Parent1[x]])
            x = self.Parent1[x]
        x = list(self.common)[0]
        while x != goal:
            path.append([x,self.Parent2[x]])
            x = self.Parent2[x]
        path = np.flip(path,axis=0)
        return path

if __name__ == '__main__':
    Astar = Weighted_A_star(0.5)
    Astar.run()
