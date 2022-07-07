#!/usr/bin/env python3
import os
import json
from collections import defaultdict
       
dic_id = []
for _, _, arquivos in os.walk("./dic"): print("")

for arquivo in arquivos:
    if arquivo.strip("_")[0] == 'd':
        dic_id.append(arquivo)


for i in range(len(dic_id)):
    print(dic_id[i])
    f = open("./dic/"+dic_id[i],)
    dct = json.load(f)

    _dct = defaultdict(list)

    for key_start in dct.keys():
        if isinstance(dct[key_start], list):
            for data in dct[key_start]:

                 if int(data) not in _dct[int(key_start)]:
                    _dct[int(key_start)].append(data)
                    _dct[data].append(int(key_start))
        else:
            if int(key_start) not in _dct[int(dct[key_start])]:
                _dct[int(dct[key_start])].append(int(key_start))
                _dct[int(key_start)].append(int(dct[key_start]))

    #print(dct)
    #print(_dct)

    f.close()

    with open("./new_dic/"+dic_id[i].split(".")[0]+".json", "w") as fp:
        json.dump(_dct, fp)
        
    print("Saved_DIC")