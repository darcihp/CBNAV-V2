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

from sklearn.cluster import KMeans

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

        img = Image.open("./png/"+self.image)
        w = img.width
        h = img.height

        print(w)
        print(h)

        img = imread("./png/"+self.image)
        img_gray = rgb2gray(img)

        image = img_as_ubyte(img_gray)

        markers = rank.gradient(image, disk(100)) < 2
        
        markers = ndi.label(markers)[0]

        gradient = rank.gradient(image, disk(5))

        labels = watershed(gradient, markers)

        '''
        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        ax[0].set_title("Original")

        ax[1].imshow(gradient, cmap=plt.cm.Spectral, interpolation='nearest')
        ax[1].set_title("Local Gradient")

        ax[2].imshow(markers, cmap=plt.cm.Spectral, interpolation='nearest')
        ax[2].set_title("Markers")

        ax[3].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        my_img = ax[3].imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest', alpha=.2)
        ax[3].set_title("Segmented")
        '''

        #https://dpi.lv/
        dpi = 166

        fig = plt.figure(frameon=False)
        fig.set_size_inches(image.shape[1]/dpi, image.shape[0]/dpi)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        ax.imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest', alpha=.2)

        plt.savefig('teste.png', dpi=dpi)

        pic = plt.imread('teste.png')/255

        img = Image.open('teste.png')
        w = img.width
        h = img.height

        self.map_image = ax

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

    #_pre_image = pre_image("2da5590ac0855ae82f82df913b13ca5b.png")

    _pre_image = pre_image(maps_id[i].strip("\n"))



    _pre_image.find_contours(True)

    while True:

        #_pre_image.get_map_image().tight_layout()
        plt.show()

        break
    #break

'''
#Cria arquivo com mapas a serem lidos
for _, _, arquivos in os.walk("./png"):print("")
for arquivo in arquivos:
    f = open("todo", "a")
    f.write(arquivo)
    f.write("\n")
    f.close()

'''