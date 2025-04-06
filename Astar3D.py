
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/")
from Search_3D.env3D import env
from Search_3D.utils3D import getDist, getRay, g_Space, Heuristic, getNearest, isCollide, \
    cost, children, StateSpace, heuristic_fun
from Search_3D.plot_util3D import visualization
import queue
import time

class Weighted_A_star(object):
    def __init__(self, resolution=0.5):
        self.Alldirec = {(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1, \
                        (-1, 0, 0): 1, (0, -1, 0): 1, (0, 0, -1): 1, \
                        (1, 1, 0): np.sqrt(2), (1, 0, 1): np.sqrt(2), (0, 1, 1): np.sqrt(2), \
                        (-1, -1, 0): np.sqrt(2), (-1, 0, -1): np.sqrt(2), (0, -1, -1): np.sqrt(2), \
                        (1, -1, 0): np.sqrt(2), (-1, 1, 0): np.sqrt(2), (1, 0, -1): np.sqrt(2), \
                        (-1, 0, 1): np.sqrt(2), (0, 1, -1): np.sqrt(2), (0, -1, 1): np.sqrt(2), \
                        (1, 1, 1): np.sqrt(3), (-1, -1, -1) : np.sqrt(3), \
                        (1, -1, -1): np.sqrt(3), (-1, 1, -1): np.sqrt(3), (-1, -1, 1): np.sqrt(3), \
                        (1, 1, -1): np.sqrt(3), (1, -1, 1): np.sqrt(3), (-1, 1, 1): np.sqrt(3)}
        self.settings = 'NonCollisionChecking'             
        self.env = env(resolution=resolution)
        self.start, self.goal = tuple(self.env.start), tuple(self.env.goal)
        self.g = {self.start:0,self.goal:np.inf}
        self.Parent = {}
        self.CLOSED = set()
        self.V = []
        self.done = False
        self.Path = []
        self.ind = 0
        self.x0, self.xt = self.start, self.goal
        self.OPEN = queue.MinheapPQ() 
        self.OPEN.put(self.x0, self.g[self.x0] + heuristic_fun(self,self.x0)) 
        self.lastpoint = self.x0

    def run(self, N=None):

        start_time = time.time() 
        
        xt = self.xt
        xi = self.x0
        while self.OPEN:
            xi = self.OPEN.get()
            if xi not in self.CLOSED:
                self.V.append(np.array(xi))
            self.CLOSED.add(xi)
            if getDist(xi,xt) < self.env.resolution:
                break
            for xj in children(self,xi):
                if xj not in self.g:
                    self.g[xj] = np.inf
                else:
                    pass
                a = self.g[xi] + cost(self, xi, xj)
                if a < self.g[xj]:
                    self.g[xj] = a
                    self.Parent[xj] = xi
                   
                    self.OPEN.put(xj, a + 1 * heuristic_fun(self, xj))

          
            if N:
                if len(self.CLOSED) % N == 0:
                    break
            if self.ind % 100 == 0: print('number node expanded = ' + str(len(self.V)))
            self.ind += 1

        self.lastpoint = xi
        if self.lastpoint in self.CLOSED:
            self.done = True
            self.Path = self.path()
            if N is None:
                visualization(self)
                plt.show()

            self.execution_time = time.time() - start_time 
            self.path_length = self.calculate_path_length()  
            self.path_smoothness = self.calculate_path_smoothness()
            self.path_safety = self.calculate_path_safety()
            self.path_directness = self.calculate_path_directness()
            print(f"Execution Time: {self.execution_time} seconds")

            print(f"Path Smoothness: {self.path_smoothness}")
            print(f"Path Safety: {self.path_safety}")
            print(f"Path Directness: {self.path_directness}")            
            print(f"Path Length: {self.path_length} units")    
            return True

        self.execution_time = time.time() - start_time  
        print(f"Execution Time: {self.execution_time} seconds")    

        return False


    def calculate_path_length(self):
        length = 0
        if self.Path and len(self.Path) > 1:
            for segment in self.Path:
                if len(segment) > 1: 
                    length += getDist(segment[0], segment[1])
        return length


    

    def path(self):
        path = []
        x = self.lastpoint
        start = self.x0
        while x != start:
            path.append([x, self.Parent[x]])
            x = self.Parent[x]
        return path


    def reset(self, xj):
        self.g = g_Space(self) 
        self.start = xj
        self.g[getNearest(self.g, self.start)] = 0 
        self.x0 = xj
        self.OPEN.put(self.x0, self.g[self.x0] + heuristic_fun(self,self.x0))  
        self.CLOSED = set()

       



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

        


if __name__ == '__main__':
    
    Astar = Weighted_A_star(0.5)
    sta = time.time()
    Astar.run()
    print(time.time() - sta)
