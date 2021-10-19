#!/usr/bin/env python3
import os
import json

for _, _, arquivos in os.walk("/home/darcihp/Doutorado/HouseExpo/HouseExpo/json"): print("")

max_num = 0

for arquivo in arquivos:
    f = open("/home/darcihp/Doutorado/HouseExpo/HouseExpo/json/"+arquivo,)
    _json = json.load(f)

    if max_num < int(_json["room_num"]):
        max_num = int(_json["room_num"])
        print(arquivo)
        print(max_num)  
    f.close()

data = [0 for _ in range(max_num+1)]

for arquivo in arquivos:
    f = open("/home/darcihp/Doutorado/HouseExpo/HouseExpo/json/"+arquivo,)
    _json = json.load(f)

    if(int(_json["room_num"]) < 1):
        data[1] = data[1] + 1
    else:
        data[int(_json["room_num"])] = data[int(_json["room_num"])] + 1
    f.close()


with open("out.txt", "w") as o:
    for line in data:
        print("{}".format(line), file=o)