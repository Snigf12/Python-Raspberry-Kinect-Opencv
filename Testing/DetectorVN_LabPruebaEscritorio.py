# -*- coding: utf-8 -*-
# Se importan las librerías a usar
from freenect import*
from numpy import*
from cv2 import*
import sys

#Funcion de Adquisicion RGB kinect
def frame_RGB():
    array,_ = sync_get_video()
    array = cvtColor(array,COLOR_RGB2BGR)
    return array

#Funcion para adquisicion de profundidad (depth) Kinect
def frame_depth():
    array,_ = sync_get_depth()
    #array = array.astype(uint8)
    return array

#Función que retorna imagen binaria donde lo verde es blanco
#y el resto es negro
def filtLAB_Verde(img):
    lab = cvtColor(img, COLOR_BGR2Lab)
    # pongo los valores verdes para hacer la mascara
    #verde_bajo = array([35, 110, 138]) -> im1  #verde_bajo = array([34, 97, 143]) -> im1
    #verde_alto = array([102, 136, 174]) -> im1  #verde_alto = array([126, 118, 174]) -> im1
    #verde_bajo = array([44, 111, 144]) -> im2  #verde_bajo = array([32, 108, 131]) -> im2
    #verde_alto = array([101, 128, 172]) -> im2  #verde_alto = array([77, 128, 153]) -> im2
    #verde_bajo = array([20, 126, 132]) -> im3  #verde_bajo = array([60, 86, 125]) -> im3
    #verde_alto = array([80, 136, 154]) -> im3  #verde_alto = array([250, 124, 179]) -> im3
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
    mascara = dilate(mascara,dil,iterations = 2) #aplico dilatacion
    return mascara


def filtLAB_Naranja(img):
    lab = cvtColor(img, COLOR_BGR2Lab)
    # pongo los valores de rango naranja para hacer la máscara
    #naranja_bajo = array([20, 146, 139]) -> im1
    #naranja_alto = array([76, 168, 166]) -> im1
    #naranja_bajo = array([35,147,144]) -> im2
    #naranja_alto = array([152,172,187]) -> im2
    #naranja_bajo = array([47, 136, 138]) -> im3
    #naranja_alto = array([228, 183, 194]) -> im3
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

def dibuja_circulos(circuloX,img):
    if circuloX is not None:
        #Se convierten los valores (x,y,r) de circulo a enteros
        circuloX = circuloX.astype("int")
        #print(circuloX)
        x=circuloX[0,0,0] 
        y=circuloX[0,0,1] 
        r=circuloX[0,0,2] 
        # solo tomo el primer circulo
        # Dibujo el circulo y luego otro circulo en el
        # centro de la esfera de radio 5
        circle(img, (x, y), r, (0, 255, 0), 5)
        circle(img, (x, y), 5, (0, 0, 255), -1) # -1 relleno

    return img


# loop principal
while True:
    frame = frame_RGB() #leo frame
    depth = frame_depth() #leo profundidad depth
    depth = resize(depth,(0,0),fx=0.5, fy=0.5)
    showdepth = depth.astype('uint8')
    showdepth = cvtColor(showdepth, COLOR_GRAY2BGR)
    
    
    mascaraV = resize(frame, (0,0), fx=0.5, fy=0.5)
    mascaraN = mascaraV
    frame = mascaraV
    frame = medianBlur(frame,5)
    #imwrite('frame.jpg',frame)
    
    mascaraV = filtLAB_Verde(frame)
    mascaraN = filtLAB_Naranja(frame)


    mascaraV = Laplacian(mascaraV,CV_8U) #Deteccion de bordes de la mascara
    mascaraN = Laplacian(mascaraN,CV_8U)
    
    diler = array([[0,1,0],
                   [1,1,1],
                   [0,1,0]],uint8) #Matriz para dilatacion y erosion
    
    mascaraV = dilate(mascaraV,diler,iterations = 1) #aplico dilatacion
    mascaraN = dilate(mascaraN,diler,iterations = 1) #aplico dilatacion

    #Hallo los círculos que estén en detección de bordes

    circuloV = HoughCircles(mascaraV,HOUGH_GRADIENT, 1, 40, param1=60,
                           param2=24,minRadius=0,maxRadius=0)
    circuloN = HoughCircles(mascaraN,HOUGH_GRADIENT, 1, 40, param1=60,
                            param2=24,minRadius=0,maxRadius=0)

    #Dibujo los circulos hallados
    dibuja_circulos(circuloV,frame)
    dibuja_circulos(circuloN,frame)



    #Para obtener la distancia depV y depN se utilizó la información de esta
    #página: https://openkinect.org/wiki/Imaging_Information (Agosto 18)
    #Esa regresión se le hicieron modificaciones para disminuir el error
    #hallando una aproximación de la forma 1/(Bx+C), donde x es el valor
    #en bytes obtenido por el sensor

    #Para la alineación
    cteX=9
    cteY=9  #Valores alineación RGB y Depth 320 x 240 (18/2)
    #circle(rgb, (80-cteX,50+cteY),40,(0,0,255),5)

    centimg = round(frame.shape[1]/2) #centro de la imagen donde son 0° horizontal
    centVert= round(frame.shape[0]/2) #centro vertical 0° vertical

    #Si encontro al menos un ciculo    
    if circuloV is not None:
        circuloV = circuloV.astype("int")
        xV = circuloV[0,0,0]
        xVd=xV + cteX
        yV = circuloV[0,0,1]
        yVd=yV - cteY
        verde=True
        if xVd >= frame.shape[1]:
            xVd = 319
        if yVd >= frame.shape[0]:
            yVd = 239

        #dibujo punto donde se medirá depth
        circle(showdepth,(xVd,yVd),5,(0,255,0),-1)

        #Paso el pixel de coordenadas RGB a depth
        depV = 1/(depth[yVd,xVd]*(-0.0028642) + 3.15221) #para obtener dato es en coordenada (y,x)->(480x640)
        depV = round(depV,4) #cuatro cifras decimales
        if depV < 0:
            depV=0
        #depV = ((4-0.8)/2048)*(depth[xVd,yVd]+1)+0.8 aprox propia
        putText(frame,str(depV)+'m', (xV,yV), FONT_HERSHEY_PLAIN, 1.5, (0,255,0),2)
    else:
        verde = False
    
    if circuloN is not None:
        circuloN = circuloN.astype("int")
        xN = circuloN[0,0,0]
        xNd=xN + cteX
        yN = circuloN[0,0,1]
        yNd=yN - cteY
        naranja=True
        if xNd >= frame.shape[1]:
            xNd = 319
        if yNd >= frame.shape[0]:
            yNd = 239
        #dibujo punto donde se medirá depth
        circle(showdepth,(xNd,yNd),5,(255,0,0),-1)

        #para obtener dato es en coordenada (y,x)->(480x640)
        depN = 1/(depth[yNd,xNd]*(-0.0028642) + 3.15221)
        depN = round(depN,4) #cuatro cifras decimales
        if depN < 0:
            depN=0
        #dep = ((4-0.8)/2048)*(depth[xNd,yNd]+1)+0.8 aprox propia
        putText(frame,str(depN)+'m', (xN,yN), FONT_HERSHEY_PLAIN, 1.5, (0,0,255),2)
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
        putText(frame,str((alphaN*180)/pi)+'grados', (50,50), FONT_HERSHEY_PLAIN, 1.5, (255,0,255),2)
    elif verde and (not naranja):
        c1,c2=0,1
        bethaV = abs(centVert - yVd)*0.17916 #0.17916 son °/Px en vertical (43°/240)
        bethaV = (bethaV*pi)/180
        depV = depV*cos(bethaV) # centro valor vertical para ubicar la distancia en 0° Vertical
        alphaV = (xVd - centimg)*0.1781 #0.1781 son los grados por pixel (°/px)
        alphaV = (alphaV*pi)/180 # en radianes
        xm = depV*sin(alphaV)
        ym = depV*cos(alphaV)
        putText(frame,str((alphaV*180)/pi)+'grados', (50,50), FONT_HERSHEY_PLAIN, 1.5, (255,80,255),2)
    else:
        c1,c2=0,0
        xm,ym=0,0

    print('c1',c1,'c2',c2,'xm',xm,'ym',ym)
    imshow("Frame", frame)
    imshow("MaskVerde", mascaraV)
    imshow("MaskNaranja", mascaraN)
    imshow("Depth medido", showdepth)        
    key = waitKey(1)
    # si se oprime la tecla 'q' se sale del loop
    if key == ord("q"):
        break

# detener y cerrar ventanas
sys.exit()
destroyAllWindows()
