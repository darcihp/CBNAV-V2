import cv2
import numpy as np
import random as rng
import json

class pre_image:
    def __init__(self,_image, _erode=62, _dilate=5, _blur=1):
        self.image = _image
        self.erode = _erode
        self.dilate = _dilate
        self.blur = _blur;
        self.d = {}

    def set_erode(self, _erode):
        self.erode = _erode

    def set_dilate(self, _dilate):
        self.dilate = _dilate

    def set_blur(self, _blur):
        self.blur = _blur

    def get_erode(self):
        return self.erode

    def get_dilate(self):
        return self.dilate

    def get_blur(self):
        return self.blur

    def save_image(self, m_image, prefix):
        cv2.imwrite(prefix+self.image, m_image)
        print("Saved_IMG")

    def save_dictionary(self, prefix):
        a_file = open(prefix+self.image.split(".")[0]+".json", "w")
        json.dump(self.d, a_file)
        a_file.close()
        print("Saved_DIC")

    def pre_imagem(self, _load=False):

        image = cv2.imread(self.image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, 0)

        if _load == True:
            countours_max = 0
            erode_max = 0
            dilate_max = 0

            for i in range(1, 256):
                ret, thresh = cv2.threshold(gray, 0, 255, 0)
                thresh = cv2.erode(thresh, None, iterations=i)
                thresh = cv2.dilate(thresh, None, iterations=self.dilate)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if (len(contours) > countours_max):
                    countours_max = len(contours)
                    erode_max = i
                    print(countours_max)

            for j in range(1, 256):
                ret, thresh = cv2.threshold(gray, 0, 255, 0)
                thresh = cv2.erode(thresh, None, iterations=erode_max)
                thresh = cv2.dilate(thresh, None, iterations=j)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if (len(contours) < countours_max):
                    break
                else:
                    dilate_max = j


            self.erode = erode_max
            self.dilate = dilate_max

        ret, thresh = cv2.threshold(gray, 0, 255, 0)
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

        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
        
        square_id = ['A','B','C','D','E','F','G','H','I']

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

        ret, thresh = cv2.threshold(res, 0, 255, 0)
        doors_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        doors_contours_poly = [None]*len(doors_contours)
        cX = [None]*len(doors_contours)
        cY = [None]*len(doors_contours)

        for i, c in enumerate(doors_contours):
            doors_contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            centers[i], radius[i] = cv2.minEnclosingCircle(doors_contours_poly[i])
           
        door_id = ['DA','DB','DC','DD','DE','DF','DG','DH','DI']

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
                    if (_horizontal == True):
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
                    else:
                        if ((k[0,0] + 1 == _min_x or k[0,0] - 1 == _min_x or k[0,0] == _min_x or
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
            connections.append(l_connection)

        self.save_image(res, "D_")

        #print(connections)
        for i, connection in enumerate(connections):
            self.d[connection[0]].append(connection[1])
            self.d[connection[1]].append(connection[0])

        #print (self.d)
        self.save_dictionary("d_")

def on_change_erode(value):
    _pre_image.set_erode(value)
    show_image()

def on_change_dilate(value):
    _pre_image.set_dilate(value)
    show_image()

def on_change_blur(value):
    _pre_image.set_blur(value)
    show_image()

def show_image():
    cv2.imshow(windowName, _pre_image.pre_imagem())


maps_id = ['01003d58d5d927cfa79cf596e87295ef.png',
'01014942a89d1f39767cb3c186e0891e.png',
'010296df7caca9a6886d5ee51538f778.png',
'0102b1a5299fcd7efefabb58d89cc609.png',
'0102f77e57d56d086c86f519da6b3099.png',
'0108996080ac89b4e476dc88e5768fad.png',
'0108b6baf430602dc5d96da68ddb4d58.png',
'010909a4dfe6950a6a6d7dc1e13550ef.png',
'01092376b49403629a78609a148be569.png',
'01014942a89d1f39767cb3c186e0891e.png',
'0109413c614258a8814099751a0871b7.png',
'010296df7caca9a6886d5ee51538f778.png',
'0109fd97ca1f0002d476d07a708b8917.png',
'0102b1a5299fcd7efefabb58d89cc609.png',
'010a80bc554638b09a1aa9d7fe684aee.png',
'0102f77e57d56d086c86f519da6b3099.png',
'010cf6dd452b6ee5ec158f307c98cb1e.png',
'0108996080ac89b4e476dc88e5768fad.png',
'010e294b8cc56aa73b24f8ab0dd4f560.png'
]

for i in range(len(maps_id)):
    _pre_image = pre_image(maps_id[i])
    _pre_image.pre_imagem(True)


#for _, _, arquivos in os.walk(ml_lab1_path + "/features"): print("")

'''
01003d58d5d927cfa79cf596e87295ef.png
01014942a89d1f39767cb3c186e0891e.png
010296df7caca9a6886d5ee51538f778.png
0102b1a5299fcd7efefabb58d89cc609.png
0102f77e57d56d086c86f519da6b3099.png
0108996080ac89b4e476dc88e5768fad.png
0108b6baf430602dc5d96da68ddb4d58.png
010909a4dfe6950a6a6d7dc1e13550ef.png
01092376b49403629a78609a148be569.png
01014942a89d1f39767cb3c186e0891e.png
0109413c614258a8814099751a0871b7.png
010296df7caca9a6886d5ee51538f778.png
0109fd97ca1f0002d476d07a708b8917.png
0102b1a5299fcd7efefabb58d89cc609.png
010a80bc554638b09a1aa9d7fe684aee.png
0102f77e57d56d086c86f519da6b3099.png
010cf6dd452b6ee5ec158f307c98cb1e.png
0108996080ac89b4e476dc88e5768fad.png
010e294b8cc56aa73b24f8ab0dd4f560.png
'''
'''
_pre_image = pre_image('0109fd97ca1f0002d476d07a708b8917.png')
_pre_image.pre_imagem(True)

windowName = 'image'
trackErode = 'trackErode'
trackDilate = 'trackDilate'
trackBlur = 'trackBlur'

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.createTrackbar(trackErode, windowName, _pre_image.get_erode(), 255, on_change_erode)
cv2.createTrackbar(trackDilate, windowName, _pre_image.get_dilate(), 255, on_change_dilate)
cv2.createTrackbar(trackBlur, windowName, _pre_image.get_blur(), 5000, on_change_blur)

show_image()

while True:
    # Press Esc to exit
    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()
'''