import matplotlib.pyplot as plt
import networkx as nx
import json
import numpy as np
import os


dic_id = []
for _, _, arquivos in os.walk("./new_count_dic"): print("")

for arquivo in arquivos:
    if arquivo.strip("_")[0] == 'd':
        dic_id.append(arquivo)


for i in range(len(dic_id)):
	print(dic_id[i])


	f = open("./new_dic/"+dic_id[i],)

	dct = json.load(f)

	list1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

	vector1 = np.array(list1)

	for data in dct.keys():
		#print(len(dct[data]))
		vector1[len(dct[data])] = vector1[len(dct[data])] + 1

	np.savetxt("./new_count_dic/"+dic_id[i]+".csv", vector1,fmt='%i', delimiter=",")


