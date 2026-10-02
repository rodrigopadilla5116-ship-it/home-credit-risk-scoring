# Credit Risk Modeling y PyL Optimization (Home Credit Default Risk)

> **Autor:** Rodrigo Antonio Padilla Lara

> **Dataset:** Home Credit Default Risk (Kaggle)

> **Stack:** Python, Pandas, NumPy, Scikit-Learn, LightGBM, Matplotlib, SciPy

## Resumen 

Este proyecto contiene un pipeline de ciencia de datos y modelado de riesgo crediticio aplicado a datos masivos de originación de créditos. Además de maximizar métricas estadísticas (como AUC), se incluye el rendimiento predictivo con la rentabilidad financiera (PyL) de la institución a través de la simulación de distintas políticas de aprobación.

Se implementaron y compararon dos enfoques metodológicos:

1. **Scorecard Tradicional (Regresión Logística + Weight of Evidence + Platt Scaling):** Enfoque normativo, auditable y explicable (cumplimiento regulatorio bancario).
2. **Machine Learning Moderno (LightGBM con Early Stopping):** Enfoque de alto rendimiento que captura interacciones no lineales complejas entre variables.

## Resultados

### Desempeño Predictivo 

**LightGBM**

ROC-AUC: 0.7640

Gini: 0.5280 

Estadístico KS: 0.3978 (Detención temprana en la iteración 635 con un AUC de validación de 0.7635)


**Regresión Logística (WoE + Calibrada)**

ROC-AUC: 0.7463

### Simulación Financiera y PyL 

Evaluando el impacto financiero con supuestos estándar (LGD = 75%, Tasa de Interés = 35%, Costo de Fondo = 15%, Spread Neto = 20%):

| Política | Umbral de PD | Tasa de Aprobación (%) | Bad Rate Esperado (%) | Pérdida Total Esperada ($) | Beneficio Neto Promedio ($) |
| --- | --- | --- | --- | --- | --- |
| **Conservador** | 0.06 | 1.13% | 0.58% | $9,597,738 | $93,926 |
| **Equilibrado** | 0.10 | 4.36% | 0.94% | $64,077,646 | $84,789 |
| **Agresivo** | 0.15 | 10.92% | 1.19% | $243,777,256 | $72,283 |

> **Conclusión:** A mayor volumen de aprobación (política agresiva), la ganancia real por cada cliente cae de forma drástica porque gran parte de ese dinero cubre carteras vencidas, así las pérdidas esperadas por impago se disparan exponencialmente (a más de $243M). La estrategia óptima depende del apetito de riesgo de la institución, pero demuestra cómo al detectar con precisión estadística a los perfiles de alto riesgo, se evita que la cartera de crédito de la institución se llene de préstamos incobrables.


## Decisiones técnicas durante la ejecución del código

* Se aplicaron reglas lógicas independientes para detectar anomalías, valores nulos y textos no válidos en variables categóricas y numéricas. Se eliminaron 28 variables redundantes de infraestructura habitacional con alta concentración de valores nulos.
* La Winsorización (percentil 99) y la imputación por mediana se calcularon estrictamente sobre el conjunto de entrenamiento (Train) y se proyectaron de forma independiente hacia Validación y Test.
* Creación de ratios financieros (como apalancamiento de crédito respecto al ingreso, carga de anualidad, capacidad de pago familiar e índices de empleo).
* Para la Regresión Logística, se transformaron las variables categóricas filtrando aquellas con un IV > 0.02 y se calibraron las probabilidades posteriores mediante *Platt Scaling* para corregir sesgos de distribución.

## Estructura del Proyecto

* Verificación de esquemas, detección de anomalías y depuración de catálogos.
* Creación de ratios financieros, división estricta de conjuntos (Train/Val/Test), Winsorización e imputación de medianas.
Cálculo de Information Value, codificación WoE, escalado estándar, entrenamiento con penalización L2 y calibración.
* Configuración del clasificador con penalización por desbalanceo (is_unbalance=True), *early stopping* (50 rounds), y cálculo de métricas de discriminación (ROC-AUC, Gini, KS, y Feature Importance por Gain).
* Evaluación de políticas de crédito bajo umbrales de probabilidad de impago (PD), cálculo de Pérdida Esperada (EL = PD * LGD * EAD) y márgenes netos.

## Para la reproducción del Proyecto

Descargar el dataset *Home Credit Default Risk* desde Kaggle (específicamente el csv de application_train y HomeCredit_columns_description) y colocar los archivos CSV dentro de la carpeta que tendrá el nombre: data/.

Ejecutar el script secuencialmente.
