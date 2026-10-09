# Referencias para el parcial · Machine Learning (UTB 2026-II)

Material de repaso preparado como **referencia de estudio** a partir de la
actividad *"Árboles de decisión a mano"*, el laboratorio del árbol de
decisión y el paper de Random Forest (Kinect). Todo en **Quarto**, con
código Python (scikit-learn) ejecutable y comentado línea por línea.

## Contenido

| archivo | tema |
|---|---|
| `00_guia_instalacion.qmd` | instalación de Quarto/Python/TinyTeX, comandos de render y **migración a RStudio Cloud** |
| `01_gini_a_mano.qmd` | **el ejercicio del parcial resuelto**: las 10 versiones (Gini, cortes, árbol de 2 niveles, matriz de confusión, métricas) + plantilla `V99` para datos nuevos |
| `02_arbol_sklearn_argumentos.qmd` | laboratorio reproducido: argumentos de `DecisionTreeClassifier`, búsqueda exhaustiva a mano, `tree_`, `plot_tree`, `export_text`, `predict_proba`, efecto de cada argumento |
| `03_random_forest.qmd` | `RandomForestClassifier`: bagging, OOB, `n_estimators`, `max_features`, importancia, árbol vs bosque, lo citable del paper |
| `04_metricas.qmd` | matriz de confusión y métricas con mini-solver (pega `y_real`/`y_pred` y sale todo) |
| `05_cheatsheet.qmd` | **hoja de repaso final**: fórmulas, reglas de oro, código esencial y respuestas de las 10 versiones |
| `06_knn.qmd` | **KNN**: distancias a mano (con caso donde escalar cambia la predicción 0→1 y 1→0), `StandardScaler`, efecto de `k`, `weights`, frontera de decisión y comparación con árbol/bosque |
| `datos/cartera_curso.csv` | cartera de 400 clientes (separador `;`, decimal `,`) usada en los docs 02, 03 y 06 |

Los `.html` y `.pdf` ya renderizados acompañan cada `.qmd`.

## Renderizar

```powershell
# desde esta carpeta:
quarto render 01_gini_a_mano.qmd          # HTML + PDF
quarto render 01_gini_a_mano.qmd --to pdf
Get-ChildItem *.qmd | ForEach-Object { quarto render $_.Name }   # todo
```

## Cambios rápidos más útiles

- **Resolver otra versión de la actividad**: en `01_gini_a_mano.qmd`, bloque
  `datos`, línea `VERSION = "V01"` → cámbiala por `"V02"`…`"V10"` y renderiza.
- **Datos nuevos del parcial**: llena la plantilla `V["V99"]` del mismo
  documento y pon `VERSION = "V99"`.
- **Otro umbral/poda en el árbol**: bloque `arbol-base` del doc `02`
  (`max_depth`, `min_samples_leaf`, `ccp_alpha`).
- **Tus predicciones en métricas**: bloque `solver` del doc `04`
  (`y_real`, `y_pred`).
- **Otro cliente nuevo en KNN**: bloque `s1-sin-escalar` del doc `06`
  (`S1`, `K`); el escalado que cambia la predicción se ve solo.

## Advertencias de estudio (lo que cambia respuestas)

1. **Hoja empatada → predice 0** (regla de scikit-learn; afecta a las
   versiones V02, V04, V06, V07, V08 y V10).
2. La actividad compara **solo los 3 cortes candidatos** del enunciado;
   scikit-learn prueba **todos** los umbrales → puede elegir otra raíz.
3. `confusion_matrix(...).ravel()` devuelve **VN, FP, FN, VP**.
