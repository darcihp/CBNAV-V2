import cv2
import numpy as np
import random as rng
import json
import os

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

import imutils

class pre_image:
    def __init__(self,_image, _erode=1, _dilate=1):
        self.image = _image
        self.map_image = None
        self.erode = _erode
        self.dilate = _dilate
        self.d = {}
        self.d_p = {}
        self.connection = []
        self.l_connection = []
        self.contours = []
        self.square_id = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Y','Z']
    
    def set_erode(self, _erode):
        self.erode = _erode

    def set_dilate(self, _dilate):
        self.dilate = _dilate

    def get_erode(self):
        return self.erode

    def get_dilate(self):
        return self.dilate

    def get_map_image(self):
        return self.map_image

    def get_l_connection(self):
        return self.l_connection

    def get_connection(self):
        return self.connection

    def get_position(self):
        return self.position

    def set_l_connection(self):
        self.l_connection = []

    def get_contours(self):
        return self.contours

    def get_square_id(self):
        return self.square_id

    def save_image(self, prefix=""):
        cv2.imwrite("./img/"+prefix+self.image, self.map_image)
        print("Saved_IMG")

    def save_dictionary(self, prefix="", position=False):

        if(position == False):
            for i, connection in enumerate(self.connection):
                self.d[connection[0]].append(connection[1])
                self.d[connection[1]].append(connection[0])

        a_file = open("./dic/"+prefix+self.image.split(".")[0]+".json", "w")

        if(position == False):
            json.dump(self.d, a_file)
        else:
            json.dump(self.d_p, a_file)

        a_file.close()
        print("Saved_DIC")

    def find_contours(self, _load=False):

        image = cv2.imread("./png/"+self.image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        distance = ndimage.distance_transform_edt(thresh)

        localMax = peak_local_max(distance, indices=False, min_distance=150, labels=thresh)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-distance, markers, mask=thresh)

        for label in np.unique(labels):
            if label == 0:
                continue

            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            print(c)

            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.map_image = image

def show_image():
    cv2.imshow(windowName, _pre_image.find_contours(True))


def click(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONUP:

        contours = param.get_contours()
        cX = [None]*len(contours)
        cY = [None]*len(contours)

        for i, c in enumerate(contours):
            M = cv2.moments(c)

            if M["m00"] != 0:
                cX[i] = int(M["m10"] / M["m00"])
                cY[i] = int(M["m01"] / M["m00"])
            else:
                cX[i] = 0
                cY[i] = 0

            a = np.array((cX[i], cY[i]))
            b = np.array((x, y))

            dist = np.sqrt(np.sum(np.square(a-b)))

            if(dist < 40):
                param.get_l_connection().append(param.get_square_id()[i])
                cv2.circle(param.get_map_image(), (cX[i],cY[i]), 30, (0,255,0), 5)
                print("OK")
                break

    if event == cv2.EVENT_MBUTTONUP:
        print(param.get_l_connection())
        print(param.get_connection())

'''
maps_id = []
for _, _, arquivos in os.walk("./png"):
    maps_id.append(arquivos)
for i in range(len(maps_id[0][:])):
    try:        
        print(maps_id[0][i])
        _pre_image = pre_image(maps_id[0][i])
        _pre_image.pre_imagem(True)
    except:
        print("Error")
        f = open("./log/error", "a")
        f.write(maps_id[0][i])
        f.write("\n")
        f.close()
        continue


'''
'''
maps_id = []
for _, _, arquivos in os.walk("./png"):
    maps_id.append(arquivos)
'''

a_file = open("todo", "r")
maps_id = a_file.readlines()
a_file.close()

for i in range(len(maps_id)):
    print(maps_id[i])

    _rooms = None

    _pre_image = pre_image(maps_id[i].strip("\n"))
    windowName = 'image'  

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 900, 900) 

    _pre_image.find_contours(True)

    cv2.setMouseCallback(windowName, click, _pre_image)

    while True:
        cv2.imshow(windowName, _pre_image.get_map_image())
        # Press Esc to exit
        ch = cv2.waitKey(5)
        if ch == 27:
            break

        ###Limpa última ligação - space
        if ch == 32:
            _pre_image.set_l_connection()
        
        ###Adiciona ligação - a
        if ch == 97:
            _pre_image.get_connection().append(_pre_image.get_l_connection())
            _pre_image.set_l_connection()
            print(_pre_image.get_connection())

        ### Salva tudo - s
        if ch == 115:
            _pre_image.save_dictionary(prefix="d_")
            _pre_image.save_dictionary(prefix="p_", position=True)
            _pre_image.save_image("I_")
            with open("todo", "w") as fout:
                fout.writelines(maps_id[i+1:])
            break

    cv2.destroyAllWindows()

'''
#Cria arquivo com mapas a serem lidos
for _, _, arquivos in os.walk("./png"):print("")
for arquivo in arquivos:
    f = open("todo", "a")
    f.write(arquivo)
    f.write("\n")
    f.close()

'''