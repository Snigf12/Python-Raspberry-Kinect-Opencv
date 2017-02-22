# -*- coding: utf-8 -*-
import RPi.GPIO
from numpy import array
from buscar_pelotasVN_LaplaceLab import*
import time

# Relacion de los pines GPIO Numero 
RPi.GPIO.setmode(RPi.GPIO.BOARD)

#Configuracion de pines de salida
# Salida serial
RPi.GPIO.setup(36,RPi.GPIO.OUT)

# Listo Raspberry
RPi.GPIO.setup(38,RPi.GPIO.OUT)

#Configuracion de pines de entrada
#Puede recibir datos: Vex Arm Cortex
RPi.GPIO.setup(40,RPi.GPIO.IN)

#while (c1 is 0) and (c2 is 0):
#while True:


while True:
    # Call the vision system:
    # c1 -> bool, if True then target is orange
    # c2 -> bool, if True then target is green
    # numx -> float x distance from sensor in cm (horizontal distance)
    # numy -> float y distance from sensor in cm (depth distance)
    c1,c2,numx,numy=buscar_pelotasVN()
    #Convert to digital values
    if numy > 0:
        print('numx',numx,'numy',numy)
        #Convierto el valor entre 0 y 255 donde 2 m
        #es el máximo valor en metros de ym
        numy = int(255*numy/2)

        print('Orange',c1,'Green', c2,'numx [cm]',numx,'numy [cm]',numy)



except KeyboardInterrupt:
    RPi.GPIO.cleanup()
