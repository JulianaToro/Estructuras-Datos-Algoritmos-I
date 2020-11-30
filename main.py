import numpy as np
import pandas as pd
import sys

#se leen los datos desde la carpeta de origen(1)

#PROOF
dato= pd.read_csv('0_test_balanced_5000.csv',sep=";", keep_default_na = False)
#se convierte en matriz de tipo dataframe 
#matrix=tuple(dato)
datoMod = dato.dropna()
datoCor = dato.convert_dtypes()
subProof = datoCor.iloc[0:100, :]
"""
subProof.drop(["estu_consecutivo.1"], axis=1)
subProof.drop(["periodo"], axis=1)
subProof.drop(["estu_tipodocumento.1"], axis=1)
subProof.drop(["estu_nacionalidad.1"], axis=1)
subProof.drop(["estu_fechanacimiento.1"], axis=1)
subProof.drop(["periodo.1"], axis=1)
subProof.drop(["estu_estudiante.1"], axis=1)
subProof.drop(["estu_pais_reside.1"], axis=1)
subProof.drop(["estu_mcpio_reside.1"], axis=1)
subProof.drop(["estu_depto_reside.1", "estu_expectativas", "cole_codigo_icfes", "cole_nombre_establecimiento", "cole_nombre_sede", "cole_cod_mcpio_ubicacion", "cole_mcpio_ubicacion", "cole_cod_depto_ubicacion", "cole_depto_ubicacion", "desemp_prof"], axis=1)
"""
#TRAINING
#se leen los datos desde la carpeta de origen(2)
dato2= pd.read_csv('0_train_balanced_15000.csv',sep=";", keep_default_na = False)
#se convierte en matriz de tipo dataframe 
#matrix2=tuple(dato2)
dato2Mod = dato2.dropna()
dato2Cor = dato2.convert_dtypes()
subTra = dato2Cor.iloc[0:10000, :]
"""
subTra.drop(["estu_consecutivo.1"], axis=1)
subTra.drop(["periodo"], axis=1)
subTra.drop(["estu_tipodocumento.1"], axis=1)
subTra.drop(["estu_nacionalidad.1"], axis=1)
subTra.drop(["estu_fechanacimiento.1"], axis=1)
subTra.drop(["periodo.1"], axis=1)
subTra.drop(["estu_estudiante.1"], axis=1)
subTra.drop(["estu_pais_reside.1"], axis=1)
subTra.drop(["estu_mcpio_reside.1"], axis=1)
subTra.drop(["estu_depto_reside.1", "estu_expectativas", "cole_codigo_icfes", "cole_nombre_establecimiento", "cole_nombre_sede", "cole_cod_mcpio_ubicacion", "cole_mcpio_ubicacion", "cole_cod_depto_ubicacion", "cole_depto_ubicacion", "desemp_prof"], axis=1)
"""
#se leen los datos desde la carpeta de origen(3)
dato3= pd.read_csv('1_test_balanced_15000.csv',sep=";")
#se convierte en matriz de tipo dataframe 
#matrix3=tuple(dato3)

#se leen los datos desde la carpeta de origen(4)
dato4= pd.read_csv('1_train_balanced_45000.csv',sep=";")
#se convierte en matriz de tipo dataframe 
#matrix4=tuple(dato4)

#se leen los datos desde la carpeta de origen(5)
dato5= pd.read_csv('2_test_balanced_25000.csv',sep=";")
#se convierte en matriz de tipo dataframe 
#matrix5=tuple(dato5)

#se leen los datos desde la carpeta de origen(6)
dato6= pd.read_csv('2_train_balanced_75000.csv',sep=";")
#se convierte en matriz de tipo dataframe 
#matrix6=tuple(dato6)

#se leen los datos desde la carpeta de origen(7)
dato7= pd.read_csv('3_test_balanced_35000.csv',sep=";")
#se convierte en matriz de tipo dataframe 
#matrix7=tuple(dato7)

training_data = [dato2]

lsubProof = np.array(subProof).tolist()
lsubTra = np.array(subTra).tolist()



header = ["estu_consecutivo.1","estu_exterior","periodo","estu_tieneetnia","estu_tomo_cursopreparacion","estu_cursodocentesies","estu_cursoiesapoyoexterno","estu_cursoiesexterna","estu_simulacrotipoicfes","estu_actividadrefuerzoareas","estu_actividadrefuerzogeneric","fami_trabajolaborpadre","fami_trabajolabormadre","fami_numlibros","estu_inst_cod_departamento","estu_tipodocumento.1","estu_nacionalidad.1","estu_genero.1","estu_fechanacimiento.1","periodo.1","estu_estudiante.1","estu_pais_reside.1","estu_depto_reside.1","estu_cod_reside_depto.1","estu_mcpio_reside.1","estu_cod_reside_mcpio.1","estu_areareside","estu_valorpensioncolegio","fami_educacionpadre.1","fami_educacionmadre.1","fami_ocupacionpadre.1","fami_ocupacionmadre.1","fami_estratovivienda.1","fami_nivelsisben","fami_pisoshogar","fami_tieneinternet.1","fami_tienecomputador.1","fami_tienemicroondas","fami_tienehorno","fami_tieneautomovil.1","fami_tienedvd","fami_tiene_nevera.1","fami_tiene_celular.1","fami_telefono.1","fami_ingresofmiliarmensual","estu_trabajaactualmente","estu_antecedentes","estu_expectativas","cole_codigo_icfes","cole_cod_dane_establecimiento","cole_nombre_establecimiento","cole_genero","cole_naturaleza","cole_calendario","cole_bilingue","cole_caracter","cole_cod_dane_sede","cole_nombre_sede","cole_sede_principal","cole_area_ubicacion","cole_jornada","cole_cod_mcpio_ubicacion","cole_mcpio_ubicacion","cole_cod_depto_ubicacion","cole_depto_ubicacion","punt_lenguaje","punt_matematicas","punt_biologia","punt_quimica","punt_fisica","punt_ciencias_sociales","punt_filosofia","punt_ingles","desemp_ingles","profundiza","puntaje_prof","desemp_prof","exito"]


def Valores(rows, col): # valores de una columna en los datos(dataframe)
    print ("datos por columna")
    print (set([row[col] for row in rows]))
    return set([row[col] for row in rows])

def ContadorTipos(rows): #contar cada tipo especificado en los datos
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question:
    #la pregunta separa los datos en los hijos del nodo padre
    #'match' es para compara el valor de la característica de la prueba con el valor que tiene almacenada la pregunta

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        """
        print("example ")
        print(example)
        print("\n")
        
        print("sel")
        print(self)
        print("\n")
        """
        
        valor = example[self.column]
        if is_numeric(valor):
            return valor >= self.value
        else:
            return valor == self.value

    def __repr__(self): #Imprimir con base a los headers y a las filas
        condicion = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condicion, str(self.value))

def partition(rows, question): #divide el conjunto de datos
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows): #impureza de Gini para una fila

    counts = ContadorTipos(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def Ganancia(izq, der, current_uncertainty):  #Ganancia de Información = uncertainty - impureza de los nodos hijod
    p = float(len(izq)) / (len(izq) + len(der))
    return current_uncertainty - p * gini(izq) - (1 - p) * gini(der)

def find_best_split(rows): #mejor pregunta
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):

        valores = set([row[col] for row in rows])

        for val in valores:  # for each value

            question = Question(col, val)


            true_rows, false_rows = partition(rows, question)


            if len(true_rows) == 0 or len(false_rows) == 0: #en caso de no dividir
                continue

            gain = Ganancia(true_rows, false_rows, current_uncertainty)


            if gain > best_gain: # se escogio > y no >= para mayor exactitud del modelo
                best_gain, best_question = gain, question

    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = ContadorTipos(rows)

class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):

    gain, question = find_best_split(rows) #busca la mejorpregunta para dividir los datos
    #Caso base-> no hay ganancia de información
    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    #Recursión para el árbol
    true_branch = build_tree(true_rows)

    #Recursión en la Rama Falsa
    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


def ImprimirArbol(nodo, sp=""):

    if isinstance(nodo, Leaf):
        print (sp + "Predicción", nodo.predictions)
        return

    print (sp + str(nodo.question))

    print (sp + '--> Verdadero:')
    ImprimirArbol(nodo.true_branch, sp + "  ")
    print (sp + '--> Falso:')
    ImprimirArbol(nodo.false_branch, sp + "  ")


def classify(row, node):
    """
    print("clasificacion")
    """
    #print(isinstance(node, Leaf))
    if isinstance(node, Leaf):
        #print(node.predictions)
        return node.predictions

    """
    print(node.question)
    print(row)
    print(node.question.match(row))
    """
    if node.question.match(row):
        """
        print("en nodo")
        print(node.question.match(row))
        """
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def ImprimirHojas(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs



if __name__ == '__main__':

    print( "Crear el arbol de entrenamiento")
    #print(pd.read_csv('0_test_balanced_5000.csv',sep=";"))
    """
    print(dato2.isnull().values.any())
    print(dato.isnull().values.any())
    print("para dato2")
    print(dato2.isnull().sum())
    print("para dato")
    print(dato.isnull().sum())


    print(dato2Mod.isnull().values.any())
    print(datoMod.isnull().values.any())
    print("para dato2Mod")
    print(dato2Mod.isnull().sum())
    print("para datoMod")
    print(datoMod.isnull().sum())
    print (dato2Cor.info())
    print (datoCor.info())

    print("info subTra")
    print (subTra.info())
    print("info subProof")
    print (subProof.info())
    """
    #Esta parte es la de entrenamiento 
    my_tree = build_tree(lsubTra)
    print (my_tree)

    #print (lsubTra)
    #print (lsubProof)
    ImprimirArbol(my_tree)
    # Evaluate
    testing_data = lsubProof

    for row in testing_data:
        """
        print("row\n")
        print(row)
        print("row[-1]\n")
        print(row[-1])
        print(classify(row, my_tree))
        ImprimirHojas(classify(row, my_tree))
        """
        print ("Actual: %s. Predicted: %s" % (row[-1], ImprimirHojas(classify(row, my_tree))))
               
