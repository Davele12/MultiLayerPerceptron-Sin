
# 1. ¿Qué tipo de red es?

 Es una **red neuronal multicapa (MLP, Multi-Layer Perceptron)**.

* Tiene **una entrada** (el ángulo en radianes, `x`).
* Una **capa oculta** con **8 neuronas** y función de activación **tanh**.
* Una **capa de salida** con **1 neurona** y salida **lineal**.

En notación de arquitectura: **\[1 → 8 → 1]**.
Esto significa: 1 neurona de entrada → 8 ocultas → 1 salida.

---

# 2. ¿Cuál es la entrada y la salida?

* **Entrada**: un número real `x` en radianes (ejemplo: `1.5708` ≈ π/2).
* **Salida**: un número real que intenta aproximar `sin(x)`.

La idea es que la red aprenda la función:

$$
f(x) \approx \sin(x)
$$

---

# 3. ¿Cómo están compuestas las capas?

## a) Capa de entrada

* **Neurona**: recibe el ángulo en radianes.
* **Normalización**: el código multiplica `x` por `1/π` (`X_SCALE`).
  Esto hace que los valores estén en un rango más pequeño (entre -2 y 2 aprox.), para que la red no se sature.

$$
x_{norm} = \frac{x}{\pi}
$$

---

## b) Capa oculta (8 neuronas con `tanh`)

Cada neurona hace:

$$
a_j = \tanh(W0_j \cdot x_{norm} + b0_j)
$$

* `W0_j` es el peso de la neurona j (multiplica la entrada).
* `b0_j` es el bias (desplaza la curva).
* `tanh` es la activación, una función que va de -1 a 1 y es suave.

Esto crea 8 curvas en forma de “S” desplazadas y escaladas.

---

## c) Capa de salida (1 neurona lineal)

La salida se calcula como:

$$
y = b1 + \sum_{j=1}^{8} W1_j \cdot a_j
$$

* `a_j` son las salidas de la capa oculta.
* `W1_j` son los pesos que combinan esas salidas.
* `b1` es el bias de salida.
* No hay activación aquí → es lineal.

Así, sumando y restando esas curvas tanh, la red puede construir algo parecido a una onda seno.

---

# 4. ¿Cómo se entrena?

Cuando `TRAIN_ON_BOARD = 1`, el código activa el entrenamiento.
El proceso es:

### a) Generar dataset

El código crea 256 puntos `Xbuf[i]` uniformemente entre $-2π, 2π$.
Para cada punto, guarda la respuesta real:

$$
Ybuf[i] = \sin(Xbuf[i])
$$

---

### b) Forward pass (predicción)

La red calcula sus predicciones `Yhatbuf[i]` para cada `Xbuf[i]` con los pesos actuales.

---

### c) Calcular error (loss function)

Usa el **error cuadrático medio (MSE)**:

$$
MSE = \frac{1}{N} \sum_i (Yhat_i - Y_i)^2
$$

---

### d) Backpropagation

Para mejorar los pesos, calcula cómo afecta cada peso al error.

* **Capa de salida**:
  El error se propaga directo porque es lineal.

$$
\delta_{out} = (Yhat - Y)
$$

* **Capa oculta**:
  Usa la derivada de `tanh`:

$$
\frac{d}{dz}\tanh(z) = 1 - \tanh(z)^2
$$

Entonces:

$$
\delta_j = \delta_{out} \cdot W1_j \cdot (1 - a_j^2)
$$

---

### e) Actualización (SGD)

Con cada delta, actualiza los pesos:

$$
W \gets W - \eta \cdot \nabla W
$$

$$
b \gets b - \eta \cdot \nabla b
$$

donde $\eta = 0.01$ es la tasa de aprendizaje (`LR`).

Esto se repite `EPOCHS` veces (ejemplo 800).

---

### f) Al terminar

Imprime por Serial los **nuevos pesos y biases** en formato C.
Tú puedes copiarlos al bloque `#else` para dejar fija la red y cambiar `TRAIN_ON_BOARD = 0`.
Así ya no entrena, solo predice rápido.

---

# 5. ¿Qué representa cada arreglo?

* `W0[8]`: pesos de la capa de entrada a la oculta.
* `b0[8]`: bias de cada neurona oculta.
* `W1[8]`: pesos de la capa oculta a la salida.
* `b1`: bias de salida.

---

# 6. Resumen conceptual

* **Tipo de red**: perceptrón multicapa (MLP).
* **Entradas**: ángulos (radianes).
* **Salidas**: valores aproximados de `sin(x)`.
* **Capas**:

  * 1 entrada
  * 1 oculta de 8 neuronas con `tanh`
  * 1 salida lineal
* **Entrenamiento**:

  * Dataset generado internamente (`sin(x)`)
  * Optimización por descenso de gradiente estocástico (SGD)
  * Error: cuadrático medio (MSE).
* **Uso**:

  * Entrenar (`TRAIN_ON_BOARD=1`) → obtener nuevos pesos.
  * Fijar pesos (`TRAIN_ON_BOARD=0`) → inferencia ligera y rápida.

---

# 7. Intuición final

* Cada neurona oculta genera una “ondita” tipo `tanh`.
* Sumando varias de esas onditas con distintos pesos y signos, la salida forma una **onda sinusoidal**.
* Con pocas neuronas (8) ya es suficiente para aproximar el seno con error bajo.
* Esto mismo se usa en IA para aproximar funciones muchísimo más complejas (voz, imágenes, etc).


