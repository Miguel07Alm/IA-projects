import sklearn
from sklearn.neural_network import MLPClassifier
import random
from random import choice
from bokeh.plotting import figure
from bokeh.io import push_notebook, show, output_notebook

#-------------------Juego piedra, papel o tijeras--------------------
opciones = ["piedra", "papel", "tijeras"]
resultado = ""
IA = ""
ganador = ""
def elegir():            #Cuando devuelve 1 es que gana el j1 y si devuelve 2 gana el j2
    return choice(opciones)
def buscar_ganador(Jugador_1, Jugador_2):
    global resultado
    if Jugador_1 == Jugador_2:
        resultado= 0

    elif Jugador_1 == "piedra" and Jugador_2 == "tijeras":
        resultado= 1
    elif Jugador_1 == "papel" and Jugador_2 == "piedra":
        resultado= 1
    elif Jugador_1 == "tijeras" and Jugador_2 == "papel":
        resultado= 1
    elif Jugador_2 == "piedra" and Jugador_1 == "tijeras":
        resultado= 2
    elif Jugador_2 == "papel" and Jugador_1 == "piedra":
        resultado= 2
    elif Jugador_2 == "tijeras" and Jugador_1 == "papel":
        resultado= 2

    return resultado

Jugador1 = input("¿Cuál es tu nombre mamawebo?")

Jugador2 = input("¿Cuál es tu nombre parguela?")
if Jugador1 !="" and Jugador2 != "":
    Jugador1 = elegir()
    Jugador2 = elegir()
    print("Mamawebo: %s Parguela: %s Gana: %s " % (
        Jugador1, Jugador2, buscar_ganador(Jugador1, Jugador2)
    ))
    IA=0


if Jugador1 =="" and Jugador2 =="":
    IA=1
def str_a_lista(opcion):
    if opcion == "piedra":
        res = (1,0,0)
    elif opcion == "papel":
        res = (0,1,0)
    else:
        res = (0,0,1)
    return res
data_x = list(map(str_a_lista, ["piedra", "tijeras", "papel"]))
data_y = list(map(str_a_lista, ["papel", "piedra", "tijeras"]))

#print(data_x)
#print(data_y)

clf = MLPClassifier(verbose=False, warm_start=True)

#--------------ENTRENAMIENTO DEL MODELO------------------------------------
modelo = clf.fit([data_x[0]], [data_y[0]])
if IA == 1:
    def juega_aprende(iters=10, debug=True):
        puntuacion = {"ganadas": 0, "perdidas": 0}
    
        data_x = []
        data_y = []
    
        for i in range(iters):
            global ganador
            Jugador1 = elegir()
        
            prediccion = modelo.predict_proba([str_a_lista(Jugador1)])[0]
            if prediccion[0] >= 0.95:
                Jugador2 = opciones[0]
            elif prediccion[1] >= 0.95:
                Jugador2 = opciones[1]
            elif prediccion[2] >= 0.95:
                Jugador2 = opciones[2]

            else:
                Jugador2 = elegir()

            if debug == True:
                print("Jugador1: %s Jugador2 (modelo): %s --> %s" %
                (

                Jugador1, prediccion, Jugador2

                ))

            ganador = buscar_ganador(Jugador1, Jugador2)
            if debug == True:
                print("Comprobamos: j1 VS j2: %s" % ganador)
            if ganador == 2:
                data_x.append(str_a_lista(Jugador1))
                data_y.append(str_a_lista(Jugador2))

                puntuacion["ganadas"]+=1
            else:
                puntuacion["perdidas"]+=1

        return puntuacion, data_x, data_y
if IA==1:
    puntuacion, data_x, data_y = juega_aprende(1, debug=False)
    print("Puntuacion: %s %s %%" % (puntuacion, (puntuacion["ganadas"]*100/(puntuacion["ganadas"]+puntuacion["perdidas"]))))
    if len(data_x):
        modelo = modelo.partial_fit(data_x, data_y)

i = 0
historic_pct = []   #Porcentaje
while IA==1:
    i+=1
    puntuacion, data_x, data_y = juega_aprende(1000, debug=False)
    pct = (puntuacion["ganadas"]*100/(puntuacion["ganadas"]+puntuacion["perdidas"]))
    historic_pct.append(pct)
    print("Repetición: %s - puntuacion: %s %s %%" % (

    i, puntuacion, pct
    ))

    if len(data_x):
        modelo = modelo.partial_fit(data_x, data_y)
    if sum(historic_pct[-9:])==900:  #cuando sume en 9 partidas el 100% se para la función
        break

#Representación gráfica de las repeticiones
if IA==1:
    x = range(len(historic_pct))
    y = historic_pct


    grafica = figure(title="El porcentaje(%) de ML en cada repetición")
    grafica.grid.grid_line_alpha=0.3
    grafica.xaxis.axis_label = 'iter'
    grafica.yaxis.axis_label = '%'

    grafica.line(x, y)
    show(grafica)

'''Hecho por el papi Miguel qué bacaneria...'''