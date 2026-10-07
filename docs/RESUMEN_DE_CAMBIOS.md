# Qué hicimos y por qué: resumen para personas no técnicas

*Actualizado el 7 de octubre de 2026. Este documento explica en lenguaje sencillo la revisión del proyecto: qué encontramos, qué cambiamos y qué significa. Los detalles técnicos están enlazados al final.*

---

## 1. El proyecto en una frase

Un banco quiere saber **qué clientes tienen más probabilidad de contratar un producto en diciembre de 2026**. Nuestro modelo le asigna a cada cliente una probabilidad, para que el banco pueda **ordenarlos** y contactar primero a los más prometedores.

El concurso califica con el **Gini**, que mide qué tan bien ordena el modelo:
- **0** equivale a ordenar al azar.
- **1** sería un orden perfecto.
- Nuestro modelo está en **≈ 0,25**. Con estos datos, ese parece ser el techo: la información disponible predice poco.

## 2. El resultado en una línea

> El modelo nuevo ordena **un poco mejor** (Gini de 0,249 a 0,254) y, sobre todo, es **más estable y más confiable**. Lo comprobamos con varias pruebas para descartar que la mejora sea suerte.

En la práctica, si el banco contacta al **10 % de clientes** que el modelo considera más probables:

| | Clientes que contratan, de cada 100 contactados |
|---|---|
| Al azar (sin modelo) | 14,8 |
| Modelo actual | 29,7 |
| **Modelo nuevo** | **30,2** |

Los dos modelos **duplican** la eficacia de contactar al azar. El nuevo añade una mejora pequeña, de medio cliente más por cada 100 contactos.

## 3. Qué revisamos

Hicimos una auditoría completa, como la que haría un revisor externo:
1. **Los datos:** qué información hay y si tiene sentido para un banco.
2. **La forma de medir:** si las cifras que reportábamos eran fiables.
3. **Los modelos:** si eran los adecuados y si se habían entrenado bien.
4. **La robustez:** si el modelo seguiría funcionando en diciembre, un mes que nunca vio.

## 4. Qué encontramos

### 4.1 Sobre los datos

- **Solo tres datos predicen de verdad:** la **banda de riesgo** del cliente, **cuántos productos** tiene y **hace cuánto hizo su última transacción**. El resto aporta muy poco.
- **Los datos parecen simulados.** Información que en un banco real va unida aquí no lo está. Por ejemplo, tener un préstamo no tiene relación con el nivel de deuda. Por eso no conviene sacar conclusiones de negocio de esas variables.
- **Una variable "se rompe" en diciembre.** `dias_ultima_interaccion` es, de enero a noviembre, una copia cada vez más degradada de otra variable. En diciembre se vuelve completamente aleatoria. Usarla es apostar por algo que en diciembre no funcionará.

### 4.2 Sobre la forma de medir: tres errores

| Error | En palabras sencillas | Consecuencia |
|---|---|---|
| **Se elegía al ganador mirando un solo mes** | Como decidir quién es el mejor alumno con solo el último examen, aunque hubo cuatro | El ganador podía cambiar por pura suerte |
| **El margen de error estaba mal calculado** | Se decía "estamos seguros con ±0,009" cuando lo real era ±0,016 | Se mostraba más certeza de la que había |
| **Los modelos se detenían casi al azar** | El modelo aprende en pasos, y la regla para decidir cuándo parar daba 8, 29 o 141 pasos según el mes | El modelo final dependía del azar |

### 4.3 Sobre los archivos del repositorio

Un script auxiliar **rellenaba cuatro archivos de resultados con números inventados**: valores aleatorios, cifras escritas a mano o una matriz vacía. A partir de ellos se generaban cuatro gráficos. **Los retiramos**, regeneramos el que se podía calcular con datos reales (la matriz de correlación) y corregimos el script para que no vuelva a inventar valores.

## 5. Qué cambiamos

| Antes | Ahora |
|---|---|
| Ganador elegido con el último mes | Ganador elegido con el **promedio de los cuatro meses**, viendo también el peor mes |
| Margen de error mal calculado | Margen de error calculado correctamente |
| El modelo se detenía casi al azar | Se elige el tamaño del modelo con una regla estable, mirando los cuatro meses a la vez |
| Árboles de decisión "grandes" (31 grupos por árbol) | Árboles **"superficiales"** (8 grupos, al menos 200 clientes por grupo) |
| Usaba la variable que se rompe en diciembre | **Se excluye** esa variable |

**¿Por qué árboles más pequeños?** Cada "árbol" del modelo divide a los clientes en grupos. Con árboles grandes, algunos grupos son tan pequeños que el modelo aprende casualidades: un grupo de 100 clientes que contrató más por azar. Como aquí la información útil es poca y simple, los árboles pequeños aprenden lo importante sin memorizar ruido.

## 6. Cómo sabemos que la mejora no es suerte

Probamos 37 versiones distintas del modelo. Cuando se prueban muchas opciones, siempre hay alguna que "gana" por casualidad. Por eso hicimos pruebas específicas:

| Pregunta | Prueba | Respuesta |
|---|---|---|
| ¿El modelo aguanta en diciembre? | Simulamos las condiciones de diciembre en meses pasados | **Sí**, pierde menos de 0,001 |
| ¿Elegir entre 37 versiones infla el resultado? | Repetimos toda la elección usando solo información del pasado de cada mes | **No**: la inflación es de solo 0,0005 |
| ¿La mejora resiste haber probado tantas versiones? | Prueba estadística diseñada para muchas comparaciones (SPA de Hansen) | **Sí** (p = 0,025) |
| ¿Sabemos cuál de las versiones "pequeñas" es la mejor? | Varias pruebas de comparación | **No**: todas son prácticamente iguales |

**Conclusión:** los árboles pequeños son mejores que el modelo actual, y eso está bien demostrado. Cuál variante exacta es la mejor no se puede saber, así que nos quedamos con la versión central, elegida antes de hacer estas pruebas.

## 7. Lo que no sabemos

- **El resultado real de diciembre.** El concurso no nos da las respuestas de diciembre. Todo lo anterior son estimaciones con los meses de agosto a noviembre.
- **La mejora es pequeña:** +0,005, por debajo del umbral de 0,01 que el equipo usa para hablar de una "mejora clara". Es una mejora consistente, pero modesta.
- **Una parte de la lista "top" depende de la muestra.** Con otra muestra de clientes, alrededor del 17 % de los clientes del 10 % superior cambiaría. Es una limitación de los datos, no del modelo, y el modelo nuevo es el que menos cambia.

## 8. Un hallazgo para el futuro

**Cuántos meses lleva el cliente en el registro** predice por sí solo casi tanto como algunas variables. Los clientes nuevos contratan más (17 %) que los que llevan muchos meses (13 %). Hoy no se usa como variable. Es la mejora con más potencial para una próxima versión.

## 9. Estado de la entrega

- **Entrega actual:** `resultados_reportes/submission_final.csv`, sin cambios.
- **Entrega candidata (modelo nuevo):** `resultados_reportes/candidata_20261007/submission_candidata.csv`.
- La candidata **cumple la regla del equipo** para reemplazar a la actual: es mejor en promedio por más de 0,003 y no empeora en ningún mes en las condiciones de diciembre. **La decisión de reemplazarla es del equipo.**

## 10. Glosario

| Término | Significado |
|---|---|
| **Gini** | Qué tan bien ordena el modelo a los clientes: 0 = azar, 1 = perfecto |
| **Modelo / árbol de decisión** | Reglas del tipo "si la banda de riesgo es baja y tiene 3 o más productos, probabilidad alta". El modelo suma cientos de árboles |
| **LightGBM, XGBoost, CatBoost** | Programas que construyen esos conjuntos de árboles |
| **Validación walk-forward** | Entrenar con los meses anteriores y probar con el mes siguiente, como pasará en diciembre |
| **Fold** | Cada uno de los meses de prueba (agosto, septiembre, octubre y noviembre) |
| **Margen de error / intervalo de confianza** | El rango en el que probablemente está el valor real |
| **Bootstrap** | Repetir el cálculo muchas veces con muestras distintas de clientes para medir cuánto varía |
| **Sobreajuste** | Cuando el modelo aprende casualidades de los datos en lugar de patrones reales |
| **Sesgo de selección** | La ventaja aparente que obtiene "el mejor de muchos" solo por haber probado muchos |

## 11. Para quien quiera más detalle

| Documento | Contenido |
|---|---|
| [`README.md`](../README.md) | Cómo está organizado el proyecto y cómo ejecutarlo |
| [`validacion_modelos/INFORME_VALIDACION.md`](../validacion_modelos/INFORME_VALIDACION.md) | ¿Se cae el modelo en diciembre? |
| [`validacion_modelos/INFORME_ROBUSTEZ.md`](../validacion_modelos/INFORME_ROBUSTEZ.md) | Nueve pruebas de robustez de los candidatos |
| [`validacion_modelos/INFORME_SOBREOPTIMIZACION.md`](../validacion_modelos/INFORME_SOBREOPTIMIZACION.md) | ¿La mejora es real o es suerte por probar muchas versiones? |
| [`resultados_reportes/candidata_20261007/reporte_modelo_lightgbm_superficial.html`](../resultados_reportes/candidata_20261007/reporte_modelo_lightgbm_superficial.html) | Ficha del modelo candidato |
| [`docs/DATASET_DESCRIPTION.md`](DATASET_DESCRIPTION.md) | Diccionario de datos |
