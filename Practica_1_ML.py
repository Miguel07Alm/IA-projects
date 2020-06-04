import numpy
from sklearn import tree

atributos = [[140, 1], [130, 1], [150, 0], [170, 0]]   #1 suave y 0 desigual
etiqueta =[0, 0, 1, 1]    # 0 MiniSandia y 1 Piña
 
clasificador = tree.DecisionTreeClassifier()  #Resultado

clasificador = clasificador.fit(atributos, etiqueta)  #Patrones

print(clasificador.predict([[150,0]]))#clasifica