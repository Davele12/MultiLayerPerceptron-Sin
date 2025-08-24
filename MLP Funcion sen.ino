/*
  MLP-Sine (Nano 33 BLE) — Versión didáctica y compacta
  -----------------------------------------------------
  - Red: 1 -> H -> 1 (H=16 por defecto), activación tanh en oculta, salida lineal.
  - Objetivo: aproximar sin(x) en [-2π, 2π].
  - Entrenamiento en placa (opcional): mini-batch SGD + datos aleatorios por época.
  - Visualización: Serial Plotter (etiquetas "nn:" y "sin:").

  Cómo usar:
  1) Pon TRAIN_ON_BOARD = 1 para entrenar en la placa (espera a que termine).
  2) Copia los pesos impresos en formato C.
  3) Cambia TRAIN_ON_BOARD = 0 y pega los pesos en la sección #else (modo inferencia).
  4) Abre Herramientas -> Serial Plotter (115200) para ver nn vs sin en tiempo real.
*/

#include <Arduino.h>
#include <math.h>
#include <stdlib.h>   // rand(), srand()

// ===================== PARÁMETROS PRINCIPALES =====================
#define TRAIN_ON_BOARD 1   // 1: entrena y luego imprime pesos | 0: solo inferencia

// --- Red y entrenamiento ---
const int   H       = 16;      // # neuronas ocultas (sube si quieres más precisión)
const int   N_TRAIN = 512;     // tamaño del dataset por época (muestras aleatorias)
const int   EPOCHS  = 2000;    // épocas de entrenamiento (más = mejor, tarda más)
const int   BATCH   = 32;      // mini-batch (32/64 suelen ir bien)
float       LR      = 0.005f;  // tasa de aprendizaje inicial (bájala si oscila)
const bool  LR_DECAY= true;    // decaer LR a la mitad a mitad de entrenamiento

// --- Visualización ---
const float PLOT_STEP = 0.05f; // paso del barrido para la gráfica (rad)
const uint16_t PLOT_DELAY_MS = 10; // pausa entre puntos (ms) en la gráfica

// --- Normalización de entrada ---
static const float X_SCALE = 1.0f / 3.14159265358979323846f; // x_norm = x / pi

// ===================== PESOS (dependen del modo) =====================
#if TRAIN_ON_BOARD
// En modo entrenamiento deben ser NO-const porque se actualizan.
static float W0[H];   // pesos entrada->oculta (1->H)
static float b0[H];   // bias capa oculta
static float W1[H];   // pesos oculta->salida (H->1)
static float b1 = 0;  // bias salida
#else
// Modo solo inferencia: pega aquí los pesos entrenados e inmutables (const).
// EJEMPLO (rellena con tus pesos reales):
static const float W0[H] = { /* pega aquí H valores */ };
static const float b0[H] = { /* pega aquí H valores */ };
static const float W1[H] = { /* pega aquí H valores */ };
static const float b1    = 0.0f;
#endif

// ===================== UTILIDADES DE LA RED =====================
static inline float act_tanh(float x){ return tanhf(x); }
static inline float tanh_deriv_from_y(float y){ return 1.0f - y*y; } // si ya tienes tanh(z), su derivada es 1-y^2

// Limpia caracteres de fin de línea en el buffer serie (útil si usas modo interactivo)
static inline void flushLineEndings(){
  while (Serial.peek()=='\n' || Serial.peek()=='\r' || Serial.peek()==' ') Serial.read();
}

// ===================== INFERENCIA =====================
// Dado x (radianes), devuelve la aproximación nn(x) ~ sin(x)
float nn_predict(float x_rad) {
  float x = x_rad * X_SCALE;   // normalización: ayuda a que tanh no se sature
  float a1[H];

  // Capa oculta: a_j = tanh( W0_j * x + b0_j )
  for (int j=0; j<H; ++j){
    float z = W0[j]*x + b0[j];
    a1[j] = act_tanh(z);
  }

  // Capa salida (lineal): y = b1 + sum_j W1_j * a1_j
  float y = b1;
  for (int j=0; j<H; ++j) y += W1[j]*a1[j];
  return y;
}

// ===================== VISUALIZACIÓN (Serial Plotter) =====================
// Imprime "nn:<...>\tsin:<...>" por línea para que el Plotter trace 2 series
void plot_sweep(float x_min, float x_max, float step, uint16_t ms_delay) {
  for (float x=x_min; x<=x_max; x+=step){
    float y_nn = nn_predict(x);
    float y_rf = sinf(x);
    Serial.print("nn:");  Serial.print(y_nn, 6);
    Serial.print("\tsin:"); Serial.println(y_rf, 6);
    delay(ms_delay);
  }
}

#if TRAIN_ON_BOARD
// ===================== ENTRENAMIENTO EN PLACA =====================

// Buffers (globales para no usar VLAs)
static float Xbuf[N_TRAIN], Ybuf[N_TRAIN];   // datos de la época actual
static float A1buf[BATCH * H];               // activaciones de la capa oculta (para un batch)
static float Yhatbuf[BATCH];                 // salidas predichas (para un batch)

// Inicialización Xavier/Glorot (buena para tanh): U[-a, a], con a = sqrt(6/(fan_in+fan_out))
static float frandu(float a){ return (2.0f*((float)rand()/RAND_MAX) - 1.0f) * a; }

static void init_weights(){
  // 1 -> H
  float a0 = sqrtf(6.0f / (1.0f + (float)H));
  for (int j=0;j<H;j++){ W0[j] = frandu(a0); b0[j] = 0.0f; }

  // H -> 1
  float a1 = sqrtf(6.0f / ((float)H + 1.0f));
  for (int j=0;j<H;j++){ W1[j] = frandu(a1); }
  b1 = 0.0f;
}

// Genera N_TRAIN muestras aleatorias uniformes en [-2π, 2π]
// Esto evita "memorizar" un grid fijo y mejora la generalización.
static void make_epoch_data(){
  for (int i=0;i<N_TRAIN;i++){
    float u = (float)rand()/RAND_MAX;     // 0..1
    float x = -2*PI + u*(4*PI);           // [-2π, 2π]
    Xbuf[i] = x;
    Ybuf[i] = sinf(x);
  }
}

// Baraja in-situ Xbuf/Ybuf (Fisher–Yates)
static void shuffle_data(){
  for (int i=0;i<N_TRAIN-1;i++){
    int j = i + (rand() % (N_TRAIN - i));
    float tx=Xbuf[i]; Xbuf[i]=Xbuf[j]; Xbuf[j]=tx;
    float ty=Ybuf[i]; Ybuf[i]=Ybuf[j]; Ybuf[j]=ty;
  }
}

// Forward sobre un rango [s, s+n) del dataset, guardando activaciones en A1buf y salidas en Yhatbuf
static void forward_batch_range(int s, int n){
  for (int i=0;i<n;i++){
    float x = Xbuf[s+i] * X_SCALE;
    // oculta
    for (int j=0;j<H;j++){
      float z = W0[j]*x + b0[j];
      A1buf[i*H + j] = tanhf(z);
    }
    // salida
    float y = b1;
    for (int j=0;j<H;j++) y += W1[j]*A1buf[i*H + j];
    Yhatbuf[i] = y;
  }
}

// Una época completa de mini-batch SGD
static void sgd_epoch(float lr){
  shuffle_data();
  for (int s=0; s<N_TRAIN; s+=BATCH){
    int n = (s+BATCH<=N_TRAIN) ? BATCH : (N_TRAIN - s);
    forward_batch_range(s, n);

    // Gradientes acumulados en el batch
    float gW0[H];   for(int j=0;j<H;j++) gW0[j]=0;
    float gb0_[H];  for(int j=0;j<H;j++) gb0_[j]=0;
    float gW1[H];   for(int j=0;j<H;j++) gW1[j]=0;
    float gb1_acc = 0;

    for (int i=0;i<n;i++){
      float e = Yhatbuf[i] - Ybuf[s+i];     // dL/dyhat (MSE sin 1/2)
      gb1_acc += e;
      for (int j=0;j<H;j++) gW1[j] += e * A1buf[i*H + j];

      // backprop a oculta
      float x = Xbuf[s+i] * X_SCALE;
      for (int j=0;j<H;j++){
        float a = A1buf[i*H + j];
        float d = e * W1[j] * (1.0f - a*a); // tanh' = 1 - a^2
        gb0_[j] += d;
        gW0[j]  += d * x;                   // entrada tiene dim 1
      }
    }

    // Actualización (promedio por muestra del batch)
    float invn = 1.0f / (float)n;
    b1 -= lr * gb1_acc * invn;
    for (int j=0;j<H;j++){
      W1[j] -= lr * gW1[j] * invn;
      W0[j] -= lr * gW0[j] * invn;
      b0[j] -= lr * gb0_[j] * invn;
    }
  }
}

// Evalúa MSE en un grid para monitoreo (no afecta el entrenamiento)
static float eval_mse_grid(int NVAL=200){
  float mse=0;
  for(int i=0;i<NVAL;i++){
    float t=(float)i/(NVAL-1);
    float x=-2*PI + t*(4*PI);
    float d = nn_predict(x) - sinf(x);
    mse += d*d;
  }
  return mse / (float)NVAL;
}

// Entrenamiento completo y volcado de pesos
static void tune_on_board(){
  srand(12345);        // semilla reproducible
  init_weights();      // Xavier para tanh

  for (int e=1; e<=EPOCHS; e++){
    make_epoch_data();
    sgd_epoch(LR);

    if (LR_DECAY && e == (EPOCHS/2)) LR *= 0.5f;  // decaimiento simple

    if (e%100==0 || e==1 || e==EPOCHS){
      float mse = eval_mse_grid();
      Serial.print("epoch "); Serial.print(e);
      Serial.print("  lr=");  Serial.print(LR, 5);
      Serial.print("  mse="); Serial.println(mse, 6);
    }
  }

  // Imprime los pesos finales en formato C (pégalos en el bloque #else para modo inferencia)
  Serial.println("\n--- NUEVOS PESOS (copiar y reemplazar en modo inferencia) ---");
  Serial.print("static const float W0[H] = { ");
  for (int j=0;j<H;j++){ Serial.print(W0[j],6); Serial.print("f"); if(j<H-1) Serial.print(", "); }
  Serial.println(" };");
  Serial.print("static const float b0[H] = { ");
  for (int j=0;j<H;j++){ Serial.print(b0[j],6); Serial.print("f"); if(j<H-1) Serial.print(", "); }
  Serial.println(" };");
  Serial.print("static const float W1[H] = { ");
  for (int j=0;j<H;j++){ Serial.print(W1[j],6); Serial.print("f"); if(j<H-1) Serial.print(", "); }
  Serial.println(" };");
  Serial.print("static const float b1 = "); Serial.print(b1,6); Serial.println("f;");
  Serial.println("--- FIN ---\n");
}
#endif // TRAIN_ON_BOARD

// ===================== PROGRAMA PRINCIPAL =====================
void setup() {
  Serial.begin(115200);
  while(!Serial){}   // espera a puerto serie listo

#if TRAIN_ON_BOARD
  Serial.println("Entrenando MLP 1-16-1 para sin(x) en Nano 33 BLE...");
  tune_on_board();   // ajusta los pesos (puede tomar un rato)
#endif

  Serial.println("Abre Herramientas -> Serial Plotter (115200). Graficando nn vs sin...");
  delay(500);
}

void loop() {
  // Barrido continuo para ver la tendencia en tiempo real
  plot_sweep(-2*PI, 2*PI, PLOT_STEP, PLOT_DELAY_MS);

  // ---- (opcional) modo interactivo por Serial ----
  // if (Serial.available()){
  //   float x = Serial.parseFloat();
  //   flushLineEndings();
  //   Serial.println(nn_predict(x), 6);
  // }
}
