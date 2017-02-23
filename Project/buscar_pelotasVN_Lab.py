# -*- coding: utf-8 -*-
# Se importan las librerías a usar
from freenect import*
from numpy import*
from cv2 import*
from time import*

def buscar_pelotasVN(): #Función principal para llamar desde programa principal
                        #para transmisión serial a CORTEX
    
    #Funcion de Adquisicion RGB kinect
    def frame_RGB():
        array,_ = sync_get_video()
        array = cvtColor(array,COLOR_RGB2BGR)
        return array

    #Funcion para adquisicion de profundidad (depth) Kinect
    def frame_depth():
        array,_ = sync_get_depth()
        return array

    #Función que retorna imagen binaria donde lo verde es blanco
    #y el resto es negro
    def filtLAB_Verde(img):
        lab = cvtColor(img, COLOR_BGR2Lab)
        # pongo los valores verdes para hacer la mascara
        #verde_bajo = array([80, 132, 0]) -> im1
        #verde_alto = array([244, 153, 110]) -> im1
        #verde_bajo = array([52, 141, 21]) -> im2
        #verde_alto = array([196, 156, 94]) -> im2
        verde_bajo = array([20, 76, 132])
        verde_alto = array([240, 121, 215])

        mascara = inRange(lab, verde_bajo, verde_alto)
        
        er = ones((7,7),uint8) #matriz para erosion
    
        dil = array([[0,0,0,1,0,0,0],
                     [0,1,1,1,1,1,0],
                     [0,1,1,1,1,1,0],
                     [1,1,1,1,1,1,1],
                     [0,1,1,1,1,1,0],
                     [0,1,1,1,1,1,0],
                     [0,0,0,1,0,0,0]],uint8) #matriz para dilatacion

        mascara = erode(mascara,er,iterations = 1) #aplico erosion
        mascara = dilate(mascara,dil,iterations = 1) #aplico dilatacion
        return mascara

    def filtLAB_Naranja(img):
        lab = cvtColor(img, COLOR_BGR2Lab)
        # pongo los valores de rango naranja para hacer la máscara
        #naranja_bajo = array([51, 158, 69]) -> im1
        #naranja_alto = array([193, 202, 112]) -> im1
        #naranja_bajo = array([44, 166, 71]) -> im2
        #naranja_alto = array([170, 205, 106]) -> im2
        naranja_bajo = array([20, 136, 152])
        naranja_alto = array([235, 192, 198])

        mascara = inRange(lab, naranja_bajo, naranja_alto)
        
        er = ones((7,7),uint8) #matriz para erosion
    
        dil = array([[0,0,0,1,0,0,0],
                     [0,1,1,1,1,1,0],
                     [0,1,1,1,1,1,0],
                     [1,1,1,1,1,1,1],
                     [0,1,1,1,1,1,0],
                     [0,1,1,1,1,1,0],
                     [0,0,0,1,0,0,0]],uint8) #matriz para dilatacion
    
        # matriz para erosión y dilación
        mascara = erode(mascara,er,iterations = 1) #aplico erosión
        mascara = dilate(mascara,dil,iterations = 2)#aplico dilatacion
        return mascara



    #Variables para retornar
    #resultado=[c1, c2, x1, x2, x3, x4, x5, x6, x7, x8, y1, y2, y3, y4, y5, y6, y7, y8]
    #           Color_|________Coordenada_X_(xm)_______|_____Coordenada_Y_(ym)________|
    # Arreglo con la información que se envía de manera serial



    #Parte principal
    # init=time() #medir tiempo
    
    frame = frame_RGB() #leo frame
    depth = frame_depth() #leo profundidad depth
    depth = resize(depth,(0,0),fx=0.5, fy=0.5)
    
    mascaraV = resize(frame, (0,0), fx=0.5, fy=0.5)
    mascaraN = mascaraV
    frame = mascaraV
    frame = medianBlur(frame,3)

    color=time()
    mascaraV = filtLAB_Verde(frame)
    mascaraN = filtLAB_Naranja(frame)

    # tc=time()-color #tiempo de filtro de color

    #Encuentro los círculos que estén en detección de bordes
    circuloV = HoughCircles(mascaraV,HOUGH_GRADIENT, 1, 40, param1=60,
                           param2=24,minRadius=0,maxRadius=0)

    circuloN = HoughCircles(mascaraN,HOUGH_GRADIENT, 1, 40, param1=60,
                            param2=24,minRadius=0,maxRadius=0)

    #Para obtener la distancia depV y depN se utilizó la información de esta
    #página: https://openkinect.org/wiki/Imaging_Information (Agosto 18)
    #Esa regresión se le hicieron modificaciones para disminuir el error
    #hallando una aproximación de la forma 1/(Bx+C), donde x es el valor
    #en bytes obtenido por el sensor

    #Para la alineación
    cteX=9
    cteY=9  #Valores alineación RGB y Depth
    #circle(rgb, (80-cteX,50+cteY),40,(0,0,255),5)

    centimg = round(frame.shape[1]/2) #centro de la imagen donde son 0°
                                      #horizontal
    centVert= round(frame.shape[0]/2) #centro vertical
    
    #Si encontro al menos un ciculo
    if circuloV is not None:
        circuloV = circuloV.astype("int")
        xV = circuloV[0,0,0]
        xVd=xV + cteX
        yV = circuloV[0,0,1]
        yVd=yV + cteY
        verde=True
        if xVd >= frame.shape[1]:
            xVd = 319
        if yVd >= frame.shape[0]:
            yVd = 239
        #para obtener dato es en coordenada (y,x)->(480x640)
        depV = 1/(depth[yVd,xVd]*(-0.0028642) + 3.15221)
        depV = round(depV,4) #cuatro cifras decimales
        if depV < 0:
            depV=0
        #depV = ((4-0.8)/2048)*(depth[xVd,yVd]+1)+0.8 aprox propia
    else:
        verde = False

    if circuloN is not None:
        circuloN = circuloN.astype("int")
        xN = circuloN[0,0,0]
        xNd=xN + cteX
        yN = circuloN[0,0,1]
        yNd=yN + cteY
        naranja=True
        if xNd >= frame.shape[1]:
            xNd = 319
        if yNd >= frame.shape[0]:
            yNd = 239

        #para obtener dato es en coordenada (y,x)->(480x640)
        depN = 1/(depth[yNd,xNd]*(-0.0028642) + 3.15221)
        depN = round(depN,4) #cuatro cirfras decimales
        if depN < 0:
            depN=0
    else:
        naranja = False

    if naranja or (verde and naranja):
        c1,c2=1,0
        bethaN = abs(centVert - yNd)*0.17916 #0.17916 son °/Px en vertical (43°/240)
        bethaN = (bethaN*pi)/180
        depN = depN*cos(bethaN) # centro valor vertical para ubicar la distancia en 0° Vertical
        alphaN = (xNd - centimg)*0.1781 #0.1781 son los grados por pixel (°/px) 320 x 240
        alphaN = (alphaN*pi)/180 # en radianes
        xm = depN*sin(alphaN)
        ym = depN*cos(alphaN)
    elif verde and (not naranja):
        c1,c2=0,1
        bethaV = abs(centVert - yVd)*0.17916 #0.17916 son °/Px en vertical (43°/240)
        bethaV = (bethaV*pi)/180
        depV = depV*cos(bethaV) # centro valor vertical para ubicar la distancia en 0° Vertical
        alphaV = (xVd - centimg)*0.1781 #0.1781 son los grados por pixel (°/px)
        alphaV = (alphaV*pi)/180 # en radianes
        xm = depV*sin(alphaV)
        ym = depV*cos(alphaV)
    else:
        c1,c2=0,0
        xm,ym=0,0
    t=time()-init
##    imshow('VERDE',mascaraV)
##    waitKey(1)
##    imshow('NARANJA',mascaraN)
##    waitKey(1)
    print('FIN',t,'EDGE',te,'COLOR',tc)
    print(c1,c2,xm,ym)
    return c1,c2,xm,ym
