#!/usr/bin/env python3
import os
import json

class BFS_SP:
    def __init__(self, graph, start, goal):
        self._graph = graph
        self._start = start
        self._goal = goal


    def SHORT(self):
        explored = []
        
        queue = [[self._start]]
        
        if self._start == self._goal:
            #print("Same Node")
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
                        #print("Shortest path = ", *new_path)
                        return len(new_path)
                explored.append(node)

        #print("So sorry, but a connecting path doesn't exist :(")
        return

dic_id = []
for _, _, arquivos in os.walk("./dic"): print("")

for arquivo in arquivos:
    if arquivo.strip("_")[0] == 'd':
        dic_id.append(arquivo)

for i in range(len(dic_id)):
    print(dic_id[i])
    f = open("./dic/"+dic_id[i],)
    dct = json.load(f)

    path_complex = 0

    for key_start in dct.keys():
        for key_stop in dct.keys():
            BFS = BFS_SP(dct, key_start, key_stop)
            _path_complex = BFS.SHORT()

            if _path_complex != None:
                if path_complex < _path_complex:
                    path_complex = _path_complex

    print(path_complex)  

    f.close()

    #break