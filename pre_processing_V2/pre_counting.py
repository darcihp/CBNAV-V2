import cv2
import numpy as np
import random as rng
import json
import os

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
        cv2.imwrite("./img/"+prefix+self.image+".png", self.map_image)
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
        image = cv2.imread("./png/"+self.image+".png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 1, 0)

        if _load == True:
            countours_max = 0
            erode_max = 0
            dilate_max = 0

            for i in range(1, 200):
                ret, thresh = cv2.threshold(gray, 0, 1, 0)
                thresh = cv2.erode(thresh, None, iterations=i)
                thresh = cv2.dilate(thresh, None, iterations=self.dilate)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if (len(contours) > countours_max):
                    countours_max = len(contours)
                    erode_max = i
                    print(countours_max)
            
            for j in range(1, 200):
                ret, thresh = cv2.threshold(gray, 0, 1, 0)
                thresh = cv2.erode(thresh, None, iterations=erode_max)
                thresh = cv2.dilate(thresh, None, iterations=j)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if (len(contours) <= countours_max):
                    break
                else:
                    dilate_max = j
            
            self.erode = erode_max
            self.dilate = dilate_max
            
            ret, thresh = cv2.threshold(gray, 0, 1, 0)
            thresh = cv2.erode(thresh, None, iterations=self.erode)
            thresh = cv2.dilate(thresh, None, iterations=self.dilate)  
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        else:
            thresh = cv2.erode(thresh, None, iterations=self.erode)
            thresh = cv2.dilate(thresh, None, iterations=self.dilate)  
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        self.contours = contours

        contours_poly = [None]*len(contours)
        #boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        cX = [None]*len(contours)
        cY = [None]*len(contours)

        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            M = cv2.moments(c)

            if M["m00"] != 0:
                cX[i] = int(M["m10"] / M["m00"])
                cY[i] = int(M["m01"] / M["m00"])
            else:
                cX[i] = 0
                cY[i] = 0

        for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.drawContours(image, contours_poly, i, (0,0,255), -1)
            cv2.putText(image, self.square_id[i], (cX[i], cY[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2) 
            '''
            Created dictionary
            '''
            if (self.square_id[i] not in self.d):
                self.d[self.square_id[i]] = []
                self.d_p[self.square_id[i]] = [(cX[i], cY[i])]

        self.map_image = image

    def pre_imagem(self, _load=False):

        image = cv2.imread("./png/"+self.image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 1, 0)

        if _load == True:
            countours_max = 0        
            erode_max = 0
            dilate_max = 0

            for i in range(1, 256):
                ret, thresh = cv2.threshold(gray, 0, 1, 0)
                thresh = cv2.erode(thresh, None, iterations=i)
                thresh = cv2.dilate(thresh, None, iterations=self.dilate)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if (len(contours) > countours_max):
                    countours_max = len(contours)
                    erode_max = i
                    print(countours_max)

            for j in range(1, 256):
                ret, thresh = cv2.threshold(gray, 0, 1, 0)
                thresh = cv2.erode(thresh, None, iterations=erode_max)
                thresh = cv2.dilate(thresh, None, iterations=j)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if (len(contours) < countours_max):
                    break
                else:
                    dilate_max = j

            self.erode = erode_max
            self.dilate = dilate_max

        ret, thresh = cv2.threshold(gray, 0, 1, 0)
        thresh = cv2.erode(thresh, None, iterations=self.erode)
        thresh = cv2.dilate(thresh, None, iterations=self.dilate)        

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        cX = [None]*len(contours)
        cY = [None]*len(contours)

        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            #centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
            M = cv2.moments(c)

            #print(contours_poly[i])

            if M["m00"] != 0:
                cX[i] = int(M["m10"] / M["m00"])
                cY[i] = int(M["m01"] / M["m00"])
            else:
                cX[i] = 0
                cY[i] = 0

        #drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
        
        square_id = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Y','Z']

        for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.drawContours(image, contours_poly, i, (0,0,255), -1)
            cv2.putText(image, square_id[i], (cX[i], cY[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)      
            '''
            Created dictionary
            '''
            if (square_id[i] not in self.d):
                self.d[square_id[i]] = []

        self.save_image(image, "I_")

        '''
        Separa portas
        '''
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0,0,0], dtype=np.uint8)
        upper_white = np.array([0,0,255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        res = cv2.bitwise_and(image,image, mask= mask)

        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(res, 0, 1, 0)
        doors_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        doors_contours_poly = [None]*len(doors_contours)
        cX = [None]*len(doors_contours)
        cY = [None]*len(doors_contours)

        for i, c in enumerate(doors_contours):
            doors_contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            centers[i], radius[i] = cv2.minEnclosingCircle(doors_contours_poly[i])
           
        door_id = ['DA','DB','DC','DD','DE','DF','DG','DH','DI','DJ','DK','DL','DM','DN','DO','DP','DQ','DR','DS','DT','DU','DV','DX','DY','DZ']

        connections = []

        for i in range(len(doors_contours)):
            color = (255, 0, 0)
            cv2.drawContours(res, doors_contours_poly, i, (0,0,255), -1)            
            #cv2.putText(res, door_id[i], (cX[i], cY[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
            #cv2.putText(res, str(cX[i])+"_"+str(cY[i]), (cX[i], cY[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
            cv2.putText(res, door_id[i], (int(centers[i][0]), int(centers[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            #print(door_id[i])
            arr = np.array(doors_contours[i])
            _min = arr.min(axis=0)
            _max = arr.max(axis=0)

            _min_x = _min[0,0]
            _min_y = _min[0,1]
            _max_x = _max[0,0]
            _max_y = _max[0,1]

            '''
            print("Min X: "+str(_min_x))
            print("Min Y: "+str(_min_y))
            print("Max X: "+str(_max_x))
            print("Max Y: "+str(_max_y))            
            '''

            _diff_x = _max_x - _min_x
            _diff_y = _max_y - _min_y

            _horizontal = False

            if (_diff_x > _diff_y):
                _horizontal = True

            arr = np.array(contours_poly, dtype=object)

            l_connection = []
            for l, j in enumerate(arr):
                for k in j:
                    #if (_horizontal == True):
                    if ((k[0,1] + 1 == _min_y or k[0,1] - 1 == _min_y or k[0,1] == _min_y or
                        k[0,1] + 1 == _max_y or k[0,1] - 1 == _max_y or k[0,1] == _max_y)):
                        
                        ok_min = False
                        ok_max = False

                        for m in j:
                            if(m[0,0] <= _min_x):
                                ok_min = True
                            if(m[0,0] >= _max_x):
                                ok_max = True
                            if (ok_min == True and ok_max == True):
                                #print(square_id[l])
                                l_connection.append(square_id[l])
                                break
                        break
                    #else:
                    elif ((k[0,0] + 1 == _min_x or k[0,0] - 1 == _min_x or k[0,0] == _min_x or
                        k[0,0] + 1 == _max_x or k[0,0] - 1 == _max_x or k[0,0] == _max_x)):
                        ok_min = False
                        ok_max = False

                        for m in j:
                            if(m[0,1] <= _min_y):
                                ok_min = True
                            if(m[0,1] >= _max_y):
                                ok_max = True
                            if (ok_min == True and ok_max == True):
                                #print(square_id[l])
                                l_connection.append(square_id[l])
                                break
                        break
                    #print ("X: "+ str(k[0,0]) + " Y: " + str(k[0,1]))   
                #print(" ")
            connections.append(l_connection)

        self.save_image(res, "D_")

        print(connections)
        for i, connection in enumerate(connections):
            self.d[connection[0]].append(connection[1])
            self.d[connection[1]].append(connection[0])

        #print (self.d)
        self.save_dictionary("d_")


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