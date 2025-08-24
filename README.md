# 🧠 MLP-Sine Approximation (Arduino Nano 33 BLE)

Este proyecto implementa una **Red Neuronal Artificial (ANN)** del tipo **Perceptrón Multicapa (MLP)** en un **Arduino Nano 33 BLE** para **aproximar la función seno** en el rango `[-2π, 2π]`.

La red entrena directamente en la placa (**TinyML on-board training**) y permite visualizar sus resultados en el **Serial Plotter**.

---

## 1. Tipo de red y cómo funciona

La arquitectura usada es un **MLP 1 → H → 1**:

- **Entrada (1):** el ángulo en radianes `x`, normalizado como `x_norm = x / π`.
- **Capa oculta (H neuronas):** aplica una transformación lineal + función de activación **tanh**:
  \[
  z_j = W0_j \cdot x_{norm} + b0_j
  \]
  \[
  a_j = \tanh(z_j)
  \]
- **Capa de salida (1):** combina las activaciones ocultas linealmente:
  \[
  \hat{y} = b1 + \sum_{j=1}^H W1_j \cdot a_j
  \]

Así, la salida \(\hat{y}\) intenta aproximar a \(\sin(x)\).

---

## 2. Capas, neuronas y pesos

- **Entrada → oculta (W0, b0):**
  - Cada neurona oculta tiene un peso `W0_j` y un bias `b0_j`.
  - Esto define cómo cada neurona “ve” la entrada (escala y desplazamiento).
- **Oculta → salida (W1, b1):**
  - Cada activación oculta `a_j` se multiplica por `W1_j`.
  - Todas las contribuciones se suman junto con un bias `b1`.

**Resumen de parámetros:**
- `W0[H]`: pesos de entrada a oculta.
- `b0[H]`: bias de cada neurona oculta.
- `W1[H]`: pesos de oculta a salida.
- `b1`: bias de salida.

---

## 3. Métodos de entrenamiento y ajuste

- **Inicialización:** Xavier/Glorot, que distribuye los pesos iniciales en un rango apropiado para `tanh`.
- **Dataset:** se generan puntos aleatorios `(x, sin(x))` en `[-2π, 2π]`.
- **Forward pass:** la red calcula `y_hat` a partir de `x`.
- **Loss:** se mide el error cuadrático medio (MSE):
  \[
  L = \frac{1}{2}(\hat{y} - y)^2
  \]
- **Backpropagation:** aplica la **regla de la cadena** para obtener los gradientes:
  - Salida:
    \[
    \frac{\partial L}{\partial W1_j} = (\hat{y}-y)\,a_j
    \]
    \[
    \frac{\partial L}{\partial b1} = \hat{y}-y
    \]
  - Oculta:
    \[
    \delta_j = (\hat{y}-y)\,W1_j\,(1-a_j^2)
    \]
    \[
    \frac{\partial L}{\partial W0_j} = \delta_j \cdot x_{norm}
    \]
    \[
    \frac{\partial L}{\partial b0_j} = \delta_j
    \]
- **Optimización:** Mini-batch SGD con tasa de aprendizaje `LR` y decaimiento opcional.

---

## 4. Resultados esperados

- Tras unas **2000 épocas** con `H=16`, la red logra aproximar la función seno con error medio cuadrático bajo (MSE ≈ 0.01–0.02).  
- En el **Serial Plotter** se visualizan dos curvas:
  - `nn:` → predicción de la red neuronal.
  - `sin:` → valor real de la función seno.

Al entrenar, la curva `nn` comienza lejos de `sin`, pero con cada época se ajusta hasta coincidir casi perfectamente.

---

## 5. Ejemplo numérico: forward + backprop

Supongamos una red **1 → 2 → 1** (para simplificar cálculos).

- Entrada: \(x = \pi/2\).  
- Normalización: \(x_{norm} = 0.5\).  
- Objetivo: \(y = \sin(\pi/2) = 1.0\).

### Forward
- Neurona 1:  
  \(z_1 = 1.0 \cdot 0.5 + 0 = 0.5,\; a_1 = \tanh(0.5) \approx 0.462\)
- Neurona 2:  
  \(z_2 = -1.0 \cdot 0.5 + 0 = -0.5,\; a_2 = \tanh(-0.5) \approx -0.462\)
- Salida:  
  \(\hat{y} = 1.0 \cdot a_1 + 1.0 \cdot a_2 = 0.0\)

Error: \(E = \hat{y} - y = -1.0\).

### Backprop
- Gradiente en salida:  
  \(g_{W1} = E \cdot a,\; g_{b1} = E\)
- Gradiente en oculta:  
  \(\delta_j = E \cdot W1_j \cdot (1-a_j^2)\)  
  \(g_{W0_j} = \delta_j \cdot x_{norm},\; g_{b0_j} = \delta_j\)

### Actualización (η=0.1)
- \(W1_1 \to 1.046\), \(W1_2 \to 0.954\)  
- \(W0_1 \to 1.039\), \(W0_2 \to -0.961\)  
- \(b0_1, b0_2 \to 0.079\), \(b1 \to 0.1\)

En el siguiente paso, la predicción sube a ≈0.297 → más cerca de 1.0.

---

## 6. Cómo usar este proyecto

1. Configura en el código:
   - `TRAIN_ON_BOARD = 1` para entrenar en la placa.  
   - Ajusta `H`, `EPOCHS`, `LR` según la precisión deseada.
2. Carga el sketch en el **Arduino Nano 33 BLE**.
3. Abre **Herramientas → Serial Plotter** a **115200 baudios**.
4. Observa la curva `nn` acercarse a `sin`.
5. Cuando estés conforme, copia los pesos impresos y fija `TRAIN_ON_BOARD = 0` para usar la red en modo inferencia.

---

## 7. Aplicaciones

- Ejemplo educativo de **TinyML**.  
- Demostración de cómo una **red neuronal** puede aproximar funciones matemáticas.  
- Base para extender a otras funciones o incluso sensores reales (ejemplo: procesar datos de IMU en el Nano 33 BLE).

---
