#!/usr/bin/env python3

class BFS_SP:
    def __init__(self, graph, start, goal):
        self._graph = graph
        self._start = start
        self._goal = goal


    def SHORT(self):
        explored = []
        
        queue = [[self._start]]
        
        if self._start == self._goal:
            print("Same Node")
            return
        
        while queue:
            path = queue.pop(0)
            node = path[-1]
            
            if node not in explored:
                neighbours = self._graph[node]

                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)

                    if neighbour == self._goal:
                        print("Shortest path = ", *new_path)
                        return
                explored.append(node)

        print("So sorry, but a connecting"\
                    "path doesn't exist :(")
        return