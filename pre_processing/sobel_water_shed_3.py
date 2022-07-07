import cv2
import numpy as np
import random as rng
import json
import os

from PIL import Image
from scipy import ndimage as ndi
import  matplotlib.pyplot as plt
from skimage.morphology import watershed, disk
from skimage import data
from skimage.io import imread
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

from skimage.filters import sobel
from sklearn.cluster import KMeans
from scipy import ndimage


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

    def save_dictionary(self, prefix="", position=False):

        with open("./dic/"+prefix+self.image.split(".")[0]+".json", "w") as fp:
            json.dump(self.d, fp)

        print("Saved_DIC")

    def find_contours_sobel(self):
        img = Image.open("./png/"+self.image)
        w = img.width
        h = img.height

        print(w)
        print(h)

        img = imread("./png/"+self.image)
        img_gray = rgb2gray(img)
        elevation_map = sobel(img_gray)     
        markers = rank.gradient(img_gray, disk(100)) < 2
        markers = ndi.label(markers)[0]

        segmentation = watershed(elevation_map, markers)

        for ir, row in enumerate(img_gray):
            for ic, col in enumerate(row):
                if col == 0:
                    segmentation[ir][ic] = 0
        anterior = 1000
        relation = False

        _dict = np.array([[0, 0], [1, 1]])

        #Linha por linha
        for ir, row in enumerate(segmentation):
            for ic, col in enumerate(row):
                if col != 0:
                    if relation == False:
                        anterior = col
                        relation = True

                    if relation == True:
                        if (col != anterior):
                            _dict = np.append(_dict, [[anterior, col]], axis = 0)
                            #np.insert(d, [1, 2], axis=0)
                            #d[anterior] = col
                            #d = dict({anterior: col})
                            #self.d.append(d)
                            #print('Relation: {} : {}'.format(anterior, col))
                        anterior = col
                if col == 0:
                    relation = False

        #Coluna por coluna
        for ir, row in enumerate(segmentation.T):
            for ic, col in enumerate(row):
                if col != 0:
                    if relation == False:
                        anterior = col
                        relation = True

                    if relation == True:
                        if (col != anterior):
                            _dict = np.append(_dict, [[anterior, col]], axis = 0)
                            #np.insert(d, [1, 2], axis=0)
                            #d[anterior] = col
                            #d = dict({anterior: col})
                            #self.d.append(d)
                            #print('Relation: {} : {}'.format(anterior, col))
                        anterior = col
                if col == 0:
                    relation = False

        #print(d)

        unique, counts = np.unique(_dict, return_counts=True)

        for ir, row in enumerate(unique):
            if (ir != 0):
                cofm = ndi.measurements.center_of_mass(img_gray, markers, row)
                if(not np.isnan(np.sum(cofm))):
                    print(cofm)
                    color = (0, 0, 0)
                    cv2.putText(segmentation, str(ir), (int(cofm[1]), int(cofm[0])), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)

        print(unique)
        print(counts)

        model = np.array([[2, 1]])

        for key in unique:
            for j in unique:
                count = 0
                for row in _dict:
                    if key == row[0] and j == row[1]:
                        count = count + 1

                if (count >= 30):
                    if key in self.d:
                        if not isinstance(self.d[key], list):
                            self.d[int(key)] = [self.d[int(key)]]

                        self.d[int(key)].append(int(j))
                    else:
                        self.d[int(key)] = int(j)

                    print("{} - {} : {}".format(key, j, count))
        print(self.d)

        #plt.subplot(122)
        plt.imshow(segmentation, cmap=plt.cm.jet)
        plt.axis('off')
        #plt.subplots_adjust(**margins)

        self.save_dictionary("d_")

        dpi = 166
        plt.savefig("./img/I_"+self.image, dpi=dpi)
        print("Saved_IMG")


'''
#Cria arquivo com mapas a serem lidos
for _, _, arquivos in os.walk("./png"):print("")
for arquivo in arquivos:
    f = open("todo", "a")
    f.write(arquivo)
    f.write("\n")
    f.close()
'''

a_file = open("todo_3", "r")
maps_id = a_file.readlines()
a_file.close()

for i in range(len(maps_id)):
    print(maps_id[i])

    _rooms = None

    #_pre_image = pre_image("2da5590ac0855ae82f82df913b13ca5b.png")

    _pre_image = pre_image(maps_id[i].strip("\n"))

    #_pre_image.find_contours(True)
    _pre_image.find_contours_sobel()