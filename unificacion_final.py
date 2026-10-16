# -*- coding: utf-8 -*-
"""
=========================================================================
UNIFICACION FINAL - Machine Learning - UTB 2026-II
=========================================================================
Un SOLO archivo que une TODOS los codigos del curso. Solo pones los datos
(cambia VERSION o llena el bloque V99) y corre el recorrido completo:

  Bloque 1  MANO    : el ejercicio del parcial a mano (Gini, cortes,
                      arbol de 2 niveles, solicitantes, matriz y metricas)
                      para V01..V10 o tus propios datos (V99).
                      [une los documentos 01, 04 y el resumen del 05]
  Bloque 2  ARBOL   : DecisionTreeClassifier sobre la cartera de 400:
                      busqueda exhaustiva a mano vs .fit, tree_,
                      plot_tree, export_text, predict_proba y el efecto
                      de cada argumento. [documento 02]
  Bloque 3  BOSQUE  : RandomForestClassifier: OOB, curva de n arboles,
                      max_features, importancia, arbol vs bosque. [doc 03]
  Bloque 4  KNN     : las dos particiones (determinista y aleatoria),
                      escalado, KNN a mano vs libreria, efecto de k,
                      unidades, maldicion de la dimensionalidad y el
                      flip de S1/S2 al escalar. [documento 06]
  Bloque 5  RESUMEN : comparativa final de los modelos + cheatsheet. [doc 05]

USO DESDE LA TERMINAL (parado en esta carpeta):

    python unificacion_final.py                  # todo, con la VERSION de abajo
    python unificacion_final.py V05              # todo, version V05
    python unificacion_final.py V99              # todo, con TUS datos (bloque V99)
    python unificacion_final.py V03 --solo mano # solo el solver a mano
    python unificacion_final.py --solo arbol bosque
    python unificacion_final.py --ver           # ademas muestra las figuras
    python unificacion_final.py --k 5 --semilla 7

Para datos nuevos SOLO se toca: el bloque V99 de la seccion DATOS (abajo)
y la linea VERSION (o se pasa la version como argumento).

El recorrido deja todo en la carpeta salidas_unificado/:
  - reporte_<VERSION>.txt : todo lo impreso, listo para entregar
  - <VERSION>_01_....png  : las figuras del recorrido

Requisitos: numpy, pandas, matplotlib, scikit-learn.
Version gemela en Quarto: parcial_referencias/07_unificacion_total.qmd
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
if "--ver" not in sys.argv:
    matplotlib.use("Agg")           # guarda figuras sin abrir ventanas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

pd.set_option("display.width", 140)

# =========================================================================
# CONFIGURACION - LO UNICO QUE SE TOCA PARA EL RECORRIDO
# =========================================================================
# CAMBIA AQUI la version del ejercicio a mano: "V01" ... "V10" o "V99"
# (V99 = tus propios datos, en la seccion DATOS de abajo).
VERSION = "V01"

RUTA_CSV = ""                     # vacio -> busca solo la cartera del curso
VARS_CARTERA = ["edad", "ingreso"] # variables predictoras de la cartera
OBJ_CARTERA = "tomo_credito"      # objetivo: 1 = si tomo el credito, 0 = no
MI_SEMILLA = 0                    # particion aleatoria de KNN (2 ult. digitos de tu codigo)
K_KNN = 9                         # vecinos (impar, como en el laboratorio)
N_ARBOLES = 200                   # arboles del bosque
MAX_DEPTH_ARBOL = 2               # profundidad del arbol sklearn del bloque 2
MIN_SAMPLES_LEAF = 5              # minimo de clientes por hoja
N_REPARTOS = 20                   # repartos aleatorios en arbol vs bosque
TAMANNOS_BOSQUE = [1, 2, 5, 10, 20, 50, 100, 200, 400]  # curva de bosque
KS_KNN = list(range(1, 60, 2))    # k impares para la curva de KNN
NUEVO_CLIENTE_BOSQUE = [48, 7.2]  # (edad, ingreso) para predecir con el bosque
NUEVO_CLIENTE_KNN = [38, 7.2]     # (edad, ingreso) para predecir con KNN

AQUI = os.path.dirname(os.path.abspath(__file__))
CANDIDATOS_CSV = [
    os.path.join(AQUI, "parcial_referencias", "datos", "cartera_curso.csv"),
    os.path.join(AQUI, "datos", "cartera_curso.csv"),
    os.path.join(AQUI, "cartera_curso.csv"),
    os.path.join(AQUI, "cartera_curso (1).csv"),
]
SALIDAS = os.path.join(AQUI, "salidas_unificado")
MOSTRAR_FIG = "--ver" in sys.argv

ANCHO = 78
FIGS = []
PREFIJO = "SIN_VERSION"

# =========================================================================
# DATOS DEL EJERCICIO A MANO: LAS 10 VERSIONES DE LA ACTIVIDAD
# =========================================================================
# Cada version es un diccionario con:
#   clientes      : 10 tuplas (nombre, ingreso, razon_deuda, mora)  <- entrenamiento
#   cortes        : los 3 cortes candidatos de la Parte B
#   segundo_corte : (variable, umbral, lado) que la actividad manda usar
#   solicitantes  : tuplas (nombre, ingreso, razon_deuda) de la Parte D
#   prueba        : tuplas (nombre, ingreso, razon, mora_real) Parte E
# =========================================================================
CORTES_BASE = [("razon_deuda", 0.35), ("ingreso", 3.5), ("razon_deuda", 0.50)]

def _v(cli, sc, sol, prueba):
    """Acorta la escritura: arma el diccionario de una version."""
    return {"clientes": cli, "cortes": CORTES_BASE, "segundo_corte": sc,
            "solicitantes": sol, "prueba": prueba}

V = {
# --------------------------------------------------------------- V01
"V01": _v([("C1",2.3,0.51,1),("C2",3.2,0.16,0),("C3",2.9,0.33,0),
           ("C4",4.3,0.41,0),("C5",6.2,0.36,0),("C6",1.8,0.23,1),
           ("C7",5.6,0.55,0),("C8",1.8,0.62,1),("C9",3.2,0.34,1),
           ("C10",1.8,0.49,1)],
          ("ingreso", 2.3, "izq"),
          [("S1",2.0,0.60),("S2",6.0,0.21)],
          [("P1",2.4,0.64,1),("P2",3.3,0.36,0),("P3",5.4,0.60,1),
           ("P4",1.8,0.36,1),("P5",3.9,0.53,0),("P6",3.7,0.23,0),
           ("P7",5.5,0.22,0),("P8",5.1,0.45,0)]),
# --------------------------------------------------------------- V02
"V02": _v([("C1",2.3,0.23,0),("C2",2.0,0.51,1),("C3",3.7,0.53,1),
           ("C4",2.9,0.23,0),("C5",3.9,0.52,0),("C6",4.8,0.47,0),
           ("C7",2.1,0.56,1),("C8",3.0,0.36,0),("C9",3.0,0.47,1),
           ("C10",4.7,0.16,0)],
          ("razon_deuda", 0.36, "izq"),
          [("S1",4.1,0.44),("S2",4.5,0.27)],
          [("P1",6.3,0.33,0),("P2",1.8,0.48,1),("P3",2.1,0.51,0),
           ("P4",5.6,0.43,0),("P5",1.7,0.58,1),("P6",4.5,0.41,0),
           ("P7",4.2,0.34,0),("P8",2.9,0.24,0)]),
# --------------------------------------------------------------- V03
"V03": _v([("C1",5.1,0.46,0),("C2",3.6,0.42,0),("C3",3.6,0.62,1),
           ("C4",5.4,0.64,0),("C5",3.2,0.26,1),("C6",2.0,0.20,0),
           ("C7",3.9,0.29,0),("C8",6.3,0.62,0),("C9",2.1,0.37,1),
           ("C10",3.6,0.52,1)],
          ("ingreso", 3.6, "der"),     # OJO: en V03 el 2do corte va al lado DERECHO
          [("S1",3.6,0.43),("S2",4.7,0.61)],
          [("P1",4.9,0.36,0),("P2",5.2,0.27,0),("P3",1.8,0.18,0),
           ("P4",5.9,0.40,0),("P5",4.5,0.57,0),("P6",1.9,0.60,1),
           ("P7",5.0,0.46,0),("P8",4.4,0.57,1)]),
# --------------------------------------------------------------- V04
"V04": _v([("C1",4.0,0.61,1),("C2",2.4,0.64,1),("C3",3.2,0.26,0),
           ("C4",2.7,0.56,1),("C5",2.2,0.38,0),("C6",1.5,0.23,0),
           ("C7",3.6,0.42,0),("C8",2.1,0.50,1),("C9",6.1,0.26,0),
           ("C10",2.6,0.30,0)],
          ("ingreso", 2.1, "izq"),
          [("S1",4.0,0.35),("S2",3.7,0.54)],
          [("P1",1.8,0.36,0),("P2",5.1,0.40,0),("P3",3.6,0.19,0),
           ("P4",3.8,0.24,0),("P5",1.9,0.41,1),("P6",2.8,0.64,1),
           ("P7",5.4,0.31,0),("P8",5.8,0.42,0)]),
# --------------------------------------------------------------- V05
"V05": _v([("C1",4.3,0.59,1),("C2",1.6,0.63,1),("C3",2.8,0.46,0),
           ("C4",2.4,0.34,1),("C5",4.4,0.26,0),("C6",4.8,0.58,0),
           ("C7",5.8,0.43,0),("C8",3.5,0.20,0),("C9",1.6,0.26,1),
           ("C10",5.2,0.38,0)],
          ("ingreso", 2.4, "izq"),
          [("S1",1.9,0.24),("S2",5.0,0.54)],
          [("P1",3.7,0.33,0),("P2",6.0,0.23,0),("P3",5.6,0.24,0),
           ("P4",1.9,0.49,1),("P5",2.4,0.34,0),("P6",5.1,0.25,0),
           ("P7",2.3,0.59,1),("P8",5.3,0.20,0)]),
# --------------------------------------------------------------- V06
"V06": _v([("C1",4.5,0.57,1),("C2",5.2,0.37,0),("C3",3.3,0.63,1),
           ("C4",5.9,0.40,0),("C5",4.8,0.58,0),("C6",2.3,0.33,1),
           ("C7",3.6,0.28,0),("C8",6.1,0.22,0),("C9",3.9,0.57,1),
           ("C10",3.0,0.31,0)],
          ("ingreso", 3.0, "izq"),
          [("S1",5.7,0.52),("S2",2.5,0.28)],
          [("P1",5.2,0.48,0),("P2",1.9,0.42,1),("P3",5.9,0.51,0),
           ("P4",4.8,0.47,0),("P5",2.5,0.64,1),("P6",3.1,0.27,0),
           ("P7",4.8,0.26,0),("P8",3.0,0.59,0)]),
# --------------------------------------------------------------- V07
"V07": _v([("C1",1.9,0.63,1),("C2",6.2,0.52,1),("C3",3.3,0.59,1),
           ("C4",6.0,0.51,0),("C5",5.3,0.25,0),("C6",1.6,0.26,1),
           ("C7",6.1,0.37,0),("C8",1.5,0.57,1),("C9",6.1,0.54,0),
           ("C10",2.6,0.34,0)],
          ("ingreso", 1.9, "izq"),
          [("S1",2.3,0.39),("S2",4.7,0.21)],
          [("P1",3.1,0.37,0),("P2",2.2,0.25,0),("P3",2.9,0.61,1),
           ("P4",1.7,0.28,0),("P5",1.6,0.24,1),("P6",2.6,0.23,0),
           ("P7",4.1,0.24,1),("P8",3.0,0.33,0)]),
# --------------------------------------------------------------- V08
"V08": _v([("C1",4.8,0.38,0),("C2",5.2,0.34,0),("C3",3.5,0.51,1),
           ("C4",2.9,0.27,0),("C5",2.9,0.18,1),("C6",5.8,0.33,0),
           ("C7",3.5,0.51,1),("C8",5.3,0.37,0),("C9",4.1,0.36,0),
           ("C10",6.0,0.63,1)],
          ("ingreso", 2.9, "izq"),
          [("S1",2.0,0.20),("S2",5.3,0.50)],
          [("P1",3.5,0.47,0),("P2",4.9,0.63,0),("P3",5.6,0.19,0),
           ("P4",2.2,0.19,0),("P5",3.2,0.57,1),("P6",3.1,0.55,1),
           ("P7",3.5,0.58,0),("P8",5.6,0.17,0)]),
# --------------------------------------------------------------- V09
"V09": _v([("C1",2.3,0.45,1),("C2",4.2,0.19,0),("C3",1.9,0.16,0),
           ("C4",1.9,0.57,1),("C5",1.8,0.48,0),("C6",6.4,0.61,0),
           ("C7",1.8,0.62,1),("C8",3.4,0.49,1),("C9",3.8,0.48,0),
           ("C10",5.0,0.16,0)],
          ("razon_deuda", 0.48, "izq"),
          [("S1",2.0,0.54),("S2",5.7,0.43)],
          [("P1",4.8,0.62,0),("P2",2.2,0.28,0),("P3",5.3,0.36,0),
           ("P4",2.9,0.55,1),("P5",4.6,0.46,0),("P6",6.4,0.35,0),
           ("P7",4.8,0.25,0),("P8",4.4,0.48,1)]),
# --------------------------------------------------------------- V10
"V10": _v([("C1",3.4,0.26,0),("C2",4.0,0.59,1),("C3",6.4,0.38,0),
           ("C4",5.0,0.57,0),("C5",2.4,0.52,1),("C6",2.4,0.21,0),
           ("C7",1.8,0.59,1),("C8",3.8,0.58,1),("C9",6.4,0.37,0),
           ("C10",6.4,0.15,0)],
          ("ingreso", 3.8, "der"),     # OJO: en V10 el 2do corte va al lado DERECHO
          [("S1",4.7,0.59),("S2",2.9,0.42)],
          [("P1",4.2,0.43,0),("P2",1.9,0.50,1),("P3",3.6,0.28,0),
           ("P4",2.1,0.56,1),("P5",5.8,0.33,0),("P6",3.0,0.57,1),
           ("P7",6.1,0.56,0),("P8",6.4,0.41,0)]),
}

# -------------------------------------------------------------------------
# PLANTILLA PARA DATOS NUEVOS: quita los # , llena con los datos de tu
# parcial y pon VERSION = "V99" (o el nombre que quieras darle arriba).
# -------------------------------------------------------------------------
# V["V99"] = _v(
#     # 10 clientes del historial: (nombre, ingreso, razon_deuda, mora)
#     [("C1", 2.3, 0.51, 1), ("C2", 3.2, 0.16, 0), ("C3", 2.9, 0.33, 0),
#      ("C4", 4.3, 0.41, 0), ("C5", 6.2, 0.36, 0), ("C6", 1.8, 0.23, 1),
#      ("C7", 5.6, 0.55, 0), ("C8", 1.8, 0.62, 1), ("C9", 3.2, 0.34, 1),
#      ("C10", 1.8, 0.49, 1)],
#     # segundo corte prescrito: (variable, umbral, lado izq/der)
#     ("ingreso", 2.3, "izq"),
#     # solicitantes de la Parte D: (nombre, ingreso, razon_deuda)
#     [("S1", 2.0, 0.60), ("S2", 6.0, 0.21)],
#     # clientes de prueba (Parte E): (nombre, ingreso, razon, mora_real)
#     [("P1", 2.4, 0.64, 1), ("P2", 3.3, 0.36, 0), ("P3", 5.4, 0.60, 1),
#      ("P4", 1.8, 0.36, 1), ("P5", 3.9, 0.53, 0), ("P6", 3.7, 0.23, 0),
#      ("P7", 5.5, 0.22, 0), ("P8", 5.1, 0.45, 0)])
#
# Si los cortes candidatos del punto 3 fueran otros:
# V["V99"]["cortes"] = [("razon_deuda", 0.40), ("ingreso", 4.0),
#                       ("razon_deuda", 0.55)]
# -------------------------------------------------------------------------

# =========================================================================
# FUNCIONES DEL SOLVER A MANO (documento 01)
# =========================================================================

def gini(y):
    """Gini de un nodo: G = 2*p*(1-p), con p = proporcion de unos (mora)."""
    if len(y) == 0:
        return 0.0
    p = float(np.mean(y))
    return 2.0 * p * (1.0 - p)

def clase_mayoritaria(y):
    """Clase que predice una hoja: la mayoritaria. Empate exacto -> 0
    (la regla de scikit-learn; usar >= 0.5 aqui es el error clasico)."""
    return int(np.mean(y) > 0.5)

def evaluar_corte(df, variable, umbral):
    """Evalua UN corte candidato 'variable <= umbral' (Parte B)."""
    x = df["razon_deuda"] if variable == "razon_deuda" else df["ingreso"]
    izq = x <= umbral
    n_izq, n_der = int(izq.sum()), int((~izq).sum())
    mor_izq = int(df.loc[izq, "mora"].sum())
    mor_der = int(df.loc[~izq, "mora"].sum())
    g_izq = gini(df.loc[izq, "mora"])
    g_der = gini(df.loc[~izq, "mora"])
    g_pond = (n_izq * g_izq + n_der * g_der) / len(df)
    return {"corte": f"{variable} <= {umbral}",
            "n_izq": n_izq, "morosos_izq": mor_izq, "G_izq": g_izq,
            "n_der": n_der, "morosos_der": mor_der, "G_der": g_der,
            "G_ponderado": g_pond}

def tabla_cortes(df, cortes):
    """Tabla del punto 3 (Parte B): evalua cada corte candidato."""
    return pd.DataFrame([evaluar_corte(df, v, u) for v, u in cortes])

def construir_arbol(df, cortes, segundo_corte):
    """Arbol de DOS NIVELES: elige raiz por menor Gini ponderado y aplica
    el segundo corte prescrito sobre el lado que indica la actividad."""
    tab = tabla_cortes(df, cortes)
    i_mejor = tab["G_ponderado"].idxmin()
    var_r, u_r = cortes[i_mejor]
    var2, u2, lado2 = segundo_corte
    x_r = df["razon_deuda"] if var_r == "razon_deuda" else df["ingreso"]
    m_raiz = x_r <= u_r
    x2 = df["razon_deuda"] if var2 == "razon_deuda" else df["ingreso"]
    m_nodo2 = m_raiz if lado2 == "izq" else ~m_raiz
    if lado2 == "izq":
        hojas = {"izq_izq": m_raiz & (x2 <= u2),
                 "izq_der": m_raiz & ~(x2 <= u2),
                 "der":     ~m_raiz}
    else:
        hojas = {"izq":     m_raiz,
                 "der_izq": m_nodo2 & (x2 <= u2),
                 "der_der": m_nodo2 & ~(x2 <= u2)}
    clases = {k: clase_mayoritaria(df.loc[m, "mora"]) for k, m in hojas.items()}
    return {"tab": tab, "mejor": tab.loc[i_mejor],
            "var_raiz": var_r, "u_raiz": u_r,
            "var2": var2, "u2": u2, "lado2": lado2,
            "hojas": hojas, "clases": clases,
            "G_raiz": gini(df["mora"]), "p_raiz": float(df["mora"].mean())}

def predecir(A, ingreso, razon_deuda):
    """Baja un cliente por el arbol hasta una hoja (Parte D)."""
    x_r = razon_deuda if A["var_raiz"] == "razon_deuda" else ingreso
    if x_r <= A["u_raiz"]:
        if A["lado2"] == "izq":
            x2 = razon_deuda if A["var2"] == "razon_deuda" else ingreso
            return A["clases"]["izq_izq" if x2 <= A["u2"] else "izq_der"]
        return A["clases"]["izq"]
    else:
        if A["lado2"] == "der":
            x2 = razon_deuda if A["var2"] == "razon_deuda" else ingreso
            return A["clases"]["der_izq" if x2 <= A["u2"] else "der_der"]
        return A["clases"]["der"]

def resolver_version(ver_id, V):
    """Resuelve UNA version completa (Partes A a F) y devuelve s, A, df, prueba."""
    d = V[ver_id]
    df = pd.DataFrame(d["clientes"],
                      columns=["cliente", "ingreso", "razon_deuda", "mora"])
    A = construir_arbol(df, d["cortes"], d["segundo_corte"])
    s = {"version": ver_id, "p_raiz": A["p_raiz"], "G_raiz": A["G_raiz"]}
    s["corte_ganador"] = A["mejor"]["corte"]
    s["G_pond_raiz"] = A["mejor"]["G_ponderado"]
    s["ganancia"] = A["G_raiz"] - A["mejor"]["G_ponderado"]
    var2, u2, lado2 = d["segundo_corte"]
    x_r = df["razon_deuda"] if A["var_raiz"] == "razon_deuda" else df["ingreso"]
    m_nodo2 = (x_r <= A["u_raiz"]).to_numpy() if lado2 == "izq" \
              else ~(x_r <= A["u_raiz"]).to_numpy()
    x2 = df["razon_deuda"] if var2 == "razon_deuda" else df["ingreso"]
    m2a = m_nodo2 & (x2 <= u2).to_numpy()
    m2b = m_nodo2 & ~(x2 <= u2).to_numpy()
    mora = df["mora"].to_numpy()
    s.update({"nodo2_n": int(m_nodo2.sum()), "nodo2_morosos": int(mora[m_nodo2].sum()),
              "n2_izq": int(m2a.sum()), "n2_der": int(m2b.sum()),
              "m2a_morosos": int(mora[m2a].sum()),
              "m2b_morosos": int(mora[m2b].sum()),
              "G2_izq": gini(mora[m2a]), "G2_der": gini(mora[m2b]),
              "G2_pond": (int(m2a.sum()) * gini(mora[m2a])
                         + int(m2b.sum()) * gini(mora[m2b])) / int(m_nodo2.sum())})
    s["S"] = {nom: predecir(A, ing, raz) for nom, ing, raz in d["solicitantes"]}
    prueba = pd.DataFrame(d["prueba"],
                          columns=["cliente", "ingreso", "razon_deuda", "mora_real"])
    prueba["pred"] = [predecir(A, i, r)
                      for i, r in zip(prueba["ingreso"], prueba["razon_deuda"])]
    return s, A, df, prueba

def dibujar_arbol(A, df, titulo=""):
    """Dibuja el arbol de 2 niveles: pregunta, n, morosos y Gini en los
    nodos; clase predicha en las hojas (Parte C, punto 6)."""
    mora = df["mora"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    x_r = df["razon_deuda"] if A["var_raiz"] == "razon_deuda" else df["ingreso"]
    m_raiz = (x_r <= A["u_raiz"]).to_numpy()
    m_nodo2 = m_raiz if A["lado2"] == "izq" else ~m_raiz
    m2a = A["hojas"]["izq_izq"] if A["lado2"] == "izq" else A["hojas"]["der_izq"]
    m2b = A["hojas"]["izq_der"] if A["lado2"] == "izq" else A["hojas"]["der_der"]
    m3 = A["hojas"]["der"] if A["lado2"] == "izq" else A["hojas"]["izq"]

    def caja(x, y, contenido, color):
        ax.annotate(contenido, xy=(x, y), ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.45", fc=color, ec="#333333", lw=1.2))

    def flecha(x1, y1, x2, y2, etiqueta, dx=0.2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + 0.2, etiqueta,
                fontsize=9, fontweight="bold",
                color="#B00020" if etiqueta == "SI" else "#00695C")

    def txt_preg(m, var, u):
        return (f"{var} <= {u}?\nn = {int(m.sum())}\n"
                f"morosos = {int(mora[m].sum())}\nGini = {gini(mora[m]):.3f}")

    def txt_hoja(m):
        c = clase_mayoritaria(mora[m])
        return (f"HOJA: {'mora (1)' if c == 1 else 'paga (0)'}\n"
                f"n = {int(m.sum())}\nmorosos = {int(mora[m].sum())}")

    col_h = lambda m: "#FFCDD2" if clase_mayoritaria(mora[m]) == 1 else "#C8E6C9"
    x_int = 2.5 if A["lado2"] == "izq" else 7.5
    x_sol = 7.5 if A["lado2"] == "izq" else 2.5
    et_r_izq = "SI" if A["lado2"] == "izq" else "NO"
    et_r_der = "NO" if A["lado2"] == "izq" else "SI"
    flecha(4.2, 8.0, x_int + 0.7, 6.15, et_r_izq)
    flecha(5.8, 8.0, x_sol, 2.35, et_r_der, dx=0.3)
    flecha(x_int - 0.8, 4.7, 1.0, 2.35, "SI")
    flecha(x_int + 0.8, 4.7, 4.0, 2.35, "NO")
    caja(5.0, 8.6, txt_preg(np.ones(len(mora), dtype=bool), A["var_raiz"], A["u_raiz"]), "#FFF3CD")
    caja(x_int, 5.4, txt_preg(m_nodo2, A["var2"], A["u2"]), "#FFF3CD")
    caja(1.0, 1.6, txt_hoja(m2a), col_h(m2a))
    caja(4.0, 1.6, txt_hoja(m2b), col_h(m2b))
    caja(x_sol, 1.6, txt_hoja(m3), col_h(m3))
    if titulo:
        ax.set_title(titulo, fontsize=11)
    return fig

# =========================================================================
# UTILIDADES DE SALIDA (banner, tablas, figuras, reporte)
# =========================================================================

def ban(texto):
    """Banner grande de seccion."""
    print("\n" + "=" * ANCHO)
    print(texto)
    print("=" * ANCHO)

def sub(texto):
    """Subtitulo de seccion."""
    print("\n--- " + texto + " " + "-" * max(0, ANCHO - len(texto) - 5))

def muestra(df, dec=3, index=False):
    """Imprime un DataFrame alineado y redondeado."""
    print(df.round(dec).to_string(index=index))

def guardar(fig, nombre):
    """Guarda la figura en salidas_unificado/ con el prefijo de la version."""
    ruta = os.path.join(SALIDAS, f"{PREFIJO}_{nombre}.png")
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    print(f"[figura guardada] {os.path.basename(ruta)}")
    if MOSTRAR_FIG:
        FIGS.append(fig)
    else:
        plt.close(fig)

class Tee:
    """Escribe en la consola Y en el archivo de reporte a la vez."""
    def __init__(self, *flujos):
        self.flujos = flujos
    def write(self, s):
        for f in self.flujos:
            f.write(s)
    def flush(self):
        for f in self.flujos:
            f.flush()

def ruta_cartera():
    """Encuentra la cartera del curso (separador ';' y decimal ',')."""
    if RUTA_CSV:
        if os.path.exists(RUTA_CSV):
            return RUTA_CSV
        sys.exit(f"No encuentro el CSV: {RUTA_CSV}")
    for c in CANDIDATOS_CSV:
        if os.path.exists(c):
            return c
    sys.exit("No encuentro cartera_curso.csv. Pasa la ruta con --csv RUTA.")

def tablero(y_real, y_pred, titulo="", pos="mora (1)", neg="paga (0)"):
    """El tablero UNICO de metricas (une los docs 04 y 06): matriz de
    confusion, tasa base, exactitud, precision, recall y F1. Calcula a
    mano y verifica contra sklearn. Devuelve un diccionario con todo."""
    y_real = np.asarray(y_real); y_pred = np.asarray(y_pred)
    n = len(y_real); p = y_real.mean()
    VN, FP, FN, VP = confusion_matrix(y_true=y_real, y_pred=y_pred,
                                       labels=[0, 1]).ravel()
    ex = (VP + VN) / n
    pr = VP / (VP + FP) if VP + FP else 0.0
    rc = VP / (VP + FN) if VP + FN else 0.0
    f1 = (2 * pr * rc / (pr + rc)) if pr + rc else 0.0
    base = max(p, 1 - p)
    assert np.isclose(ex, accuracy_score(y_true=y_real, y_pred=y_pred))
    assert np.isclose(pr, precision_score(y_true=y_real, y_pred=y_pred,
                                          pos_label=1, zero_division=0))
    assert np.isclose(rc, recall_score(y_true=y_real, y_pred=y_pred,
                                       pos_label=1, zero_division=0))
    assert np.isclose(f1, f1_score(y_true=y_real, y_pred=y_pred,
                                   pos_label=1, zero_division=0))
    if titulo:
        print(f"\n{titulo}")
    print(f"                  predicho 0      predicho 1")
    print(f"    real 0            {VN:5d}             {FP:5d}    ({neg})")
    print(f"    real 1            {FN:5d}             {VP:5d}    ({pos})")
    print(f"    tasa base : {base:.3f}   <- el numero que hay que superar")
    print(f"    exactitud : {ex:.3f}    precision : {pr:.3f}")
    print(f"    recall    : {rc:.3f}    F1        : {f1:.3f}")
    return {"VN": VN, "FP": FP, "FN": FN, "VP": VP, "n": n, "base": base,
            "exactitud": ex, "precision": pr, "recall": rc, "F1": f1}

# =========================================================================
# BLOQUE 1 - EL EJERCICIO A MANO (documentos 01 y 04)
# =========================================================================

def bloque1_mano():
    ban(f"BLOQUE 1 - A MANO: el ejercicio del parcial, version {VERSION}")
    if VERSION not in V:
        sys.exit(f"La version '{VERSION}' no existe. Validas: {sorted(V)}\n"
                 "Para datos propios llena el bloque V99 de la seccion DATOS.")
    s, A, df, prueba = resolver_version(VERSION, V)

    print("\nLos 10 clientes del historial:")
    muestra(df, index=False)

    sub("Parte A - ¿Que tan revuelta esta la caja? (punto 1 y 2)")
    print(f"Morosos = {int(df['mora'].sum())} de {len(df)} clientes.")
    print(f"  p = {int(df['mora'].sum())}/{len(df)} = {s['p_raiz']:.3f}")
    print(f"  G = 2 * p * (1-p) = {s['G_raiz']:.3f}")
    print("Si los 10 cayeran en mora (p=1): G = 0 (caja pura, no hay revuelta).")
    print("Si fueran 5 y 5 (p=0.5): G = 0.5 (revuelta maxima).")

    sub("Parte B - Elegir la primera pregunta (puntos 3 y 4)")
    print("La tabla de los tres cortes candidatos (Gini ponderado):")
    muestra(A["tab"], index=False)
    otras = ", ".join(f"{r['corte']}: {r['G_ponderado']:.3f}"
                      for _, r in A["tab"].drop(A["mejor"].name).iterrows())
    print(f"\nGana '{s['corte_ganador']}' por ser el de MENOR Gini ponderado "
          f"({s['G_pond_raiz']:.3f}, contra {otras}).")
    print(f"Ganancia = G_raiz - G_part = {s['G_raiz']:.3f} - "
          f"{s['G_pond_raiz']:.3f} = {s['ganancia']:.3f}")
    print("[OK] n_izq + n_der = 10 en las tres filas y todo Gini entre 0 y 0.5.")

    sub("Parte C - El segundo nivel (punto 5)")
    lado_txt = "izquierdo (los que SI cumplen)" if A["lado2"] == "izq" \
               else "derecho (los que NO cumplen)"
    print(f"El corte ganador deja un nodo revuelto del lado {lado_txt},")
    print(f"con {s['nodo2_n']} clientes, de los cuales {s['nodo2_morosos']} son morosos.")
    print(f"La actividad manda partirlo con '{A['var2']} <= {A['u2']}':")
    print(f"  * izquierda: n = {s['n2_izq']}, morosos = {s['m2a_morosos']}, G = {s['G2_izq']:.3f}")
    print(f"  * derecha  : n = {s['n2_der']}, morosos = {s['m2b_morosos']}, G = {s['G2_der']:.3f}")
    print(f"  * Gini ponderado del nodo partido = {s['G2_pond']:.3f}")
    print("Punto 6: el dibujo del arbol completo ->")
    fig = dibujar_arbol(A, df, f"{VERSION} - Arbol completo (dos niveles)")
    guardar(fig, "01_arbol_a_mano.png")

    sub("Parte D - Usar el arbol (punto 7)")
    for nom, ing, raz in V[VERSION]["solicitantes"]:
        print(f"  {nom} (ingreso {ing}, razon {raz}): prediccion = "
              f"{'mora (1)' if s['S'][nom] == 1 else 'paga (0)'}")

    sub("Parte E - Evaluar con clientes que no vio (puntos 9, 10 y 11)")
    print("Prediccion del arbol para cada cliente de prueba:")
    muestra(prueba, index=False)
    m = tablero(prueba["mora_real"].values, prueba["pred"].values,
                titulo=f"Matriz de confusion de {VERSION} (positivo = mora)")
    s.update(m)

    sub("Parte F - Interpretar (puntos 12 y 13)")
    supera = m["exactitud"] > m["base"]
    print(f"La exactitud es {m['exactitud']:.3f} y la tasa base {m['base']:.3f}: "
          f"{'SI la supera' if supera else 'NO la supera'}.")
    if supera:
        print("El arbol aprendio un patron que acierta mas que predecir la")
        print("clase mayoritaria a ciegas.")
    else:
        print("Este arbol es PEOR que predecir siempre la clase mayoritaria:")
        print("no aprendio un patron util con estos datos.")
    print(f"Precision = {m['precision']:.3f}: de cada 100 clientes a los que el")
    print(f"  arbol mando a mora, {m['precision']*100:.0f} eran mora de verdad.")
    print(f"Recall = {m['recall']:.3f}: de cada 100 morosos reales, el arbol")
    print(f"  atrapo {m['recall']*100:.0f} (el resto se cuela como buenos pagadores).")
    print("\nPunto 8 (conceptual): el arbol NO necesita escalar porque sus")
    print("preguntas comparan una variable consigo misma; KNN si, porque su")
    print("distancia mezcla variables distintas en una sola formula. Si el")
    print("arbol creciera sin limite, memorizaria el entrenamiento")
    print("(sobreajuste) y fallaria con clientes nuevos.")
    print("Punto 14 (el negocio): el error caro del banco es el FN (prestar a")
    print("quien no paga); por eso preferiria mas recall, aceptando menos precision.")
    print("Punto 15 (del lapiz a Python):")
    print('  arbol = DecisionTreeClassifier(criterion="gini", max_depth=2,')
    print('                              min_samples_leaf=1, random_state=0)')
    print("  recall = recall_score(y_true=y_test, y_pred=y_pred, pos_label=1)")

    sub("Comprobacion con scikit-learn (extra)")
    print("sklearn prueba TODOS los umbrales (la actividad solo 3 cortes),")
    print("asi que puede elegir otra raiz; con max_depth=2 el espiritu es el mismo:")
    X = df[["ingreso", "razon_deuda"]].values
    y = df["mora"].values
    arbol_sk = DecisionTreeClassifier(criterion="gini", max_depth=2,
                                      random_state=0).fit(X, y)
    print(export_text(arbol_sk, feature_names=["ingreso", "razon_deuda"],
                      decimals=2))
    for nom, ing, raz in V[VERSION]["solicitantes"]:
        print(f"  {nom} -> sklearn predice: {arbol_sk.predict([[ing, raz]])[0]}")

    sub(f"Resumen de TODAS las versiones ({', '.join(sorted(V))})")
    filas = []
    for vid in sorted(V):
        s2, _, _, _ = resolver_version(vid, V)
        filas.append(s2)
    res = pd.DataFrame(filas)
    resABC = res[["version", "p_raiz", "G_raiz", "corte_ganador", "G_pond_raiz",
                  "ganancia", "nodo2_n", "nodo2_morosos", "n2_izq", "n2_der",
                  "G2_pond"]].round(3)
    print("Partes A, B y C (raiz, corte ganador y segundo nivel):")
    muestra(resABC, index=False)
    resDEF = res[["version"]].copy()
    resDEF["S1"] = [s2["S"]["S1"] for s2 in filas]
    resDEF["S2"] = [s2["S"]["S2"] for s2 in filas]
    resDEF = pd.concat([resDEF,
                        res[["VP", "FN", "FP", "VN", "tasa_base", "exactitud",
                             "precision", "recall", "f1"]].round(3)], axis=1)
    print("\nPartes D, E y F (solicitantes, matriz de confusion y metricas):")
    muestra(resDEF, index=False)
    for vid in sorted(V):
        tab = tabla_cortes(pd.DataFrame(V[vid]["clientes"],
                          columns=["cliente", "ingreso", "razon_deuda", "mora"]),
                          V[vid]["cortes"])
        assert (tab["n_izq"] + tab["n_der"] == 10).all()
        assert tab[["G_izq", "G_der", "G_ponderado"]].stack().between(0, 0.5).all()
    assert res[["VP", "FN", "FP", "VN"]].sum(axis=1).eq(8).all()
    print("\n[OK] Comprobaciones internas: n_izq+n_der=10 en las tres filas, todo")
    print("Gini entre 0 y 0.5, y las matrices suman 8 en todas las versiones.")
    return s

# === PARTE 2 SE ANEXA ABAJO ===
