#include <Arduino.h>
#include <math.h>

// ===== CONFIG =====
#define TRAIN_ON_BOARD 1      // 1 = entrenar y luego imprimir pesos; 0 = solo inferencia
const int   H = 8;            // neuronas ocultas
const int   N_TRAIN = 256;    // puntos de entrenamiento (grid)
const int   EPOCHS  = 800;    // sube a 1200-1500 si quieres más fino
const float LR      = 0.01f;  // tasa de aprendizaje

// Normalización: x_norm = x * (1/pi)
static const float X_SCALE = 1.0f / 3.14159265358979323846f;

// ===== PESOS (si entrenas deben ser NO-const) =====
// Puedes dejar estos como arranque; el entrenamiento los ajusta.
#if TRAIN_ON_BOARD
static float W0[8] = {  1.214231f, -0.982144f, 0.801552f, -0.651303f,
                        0.521377f, -0.419102f, 0.333942f, -0.264151f };
static float b0[8] = {  0.000000f,  0.142311f, -0.118420f,  0.096423f,
                       -0.078114f,  0.062310f, -0.049101f,  0.038220f };
static float W1[8] = {  0.957811f, -0.848113f, 0.689225f, -0.566432f,
                        0.458210f, -0.371552f, 0.292441f, -0.232118f };
static float b1    =  0.000000f;
#else
// Si solo quieres inferencia, puedes dejar pesos entrenados y marcarlos const:
static const float W0[8] = {  1.214231f, -0.982144f, 0.801552f, -0.651303f,
                              0.521377f, -0.419102f, 0.333942f, -0.264151f };
static const float b0[8] = {  0.000000f,  0.142311f, -0.118420f,  0.096423f,
                             -0.078114f,  0.062310f, -0.049101f,  0.038220f };
static const float W1[8] = {  0.957811f, -0.848113f, 0.689225f, -0.566432f,
                              0.458210f, -0.371552f, 0.292441f, -0.232118f };
static const float b1    =  0.000000f;
#endif

// ===== UTILIDADES =====
static inline float act_tanh(float x){ return tanhf(x); }
static inline float tanh_deriv_from_y(float y){ return 1.0f - y*y; }

// Limpia terminadores del monitor serie
static inline void flushLineEndings(){
  while (Serial.peek()=='\n' || Serial.peek()=='\r' || Serial.peek()==' ')
    Serial.read();
}

// ===== INFERENCIA (1 -> H -> 1) =====
float nn_predict(float x_rad) {
  float x = x_rad * X_SCALE;
  float a1[8];
  for (int j = 0; j < H; ++j) {
    float z = W0[j] * x + b0[j];
    a1[j] = act_tanh(z);
  }
  float y = b1;
  for (int j = 0; j < H; ++j) y += W1[j] * a1[j];
  return y;
}

#if TRAIN_ON_BOARD
// ===== BUFFERS DE ENTRENAMIENTO (tamaños fijos, sin VLAs) =====
static float Xbuf[N_TRAIN], Ybuf[N_TRAIN];
static float A1buf[N_TRAIN * 8];
static float Yhatbuf[N_TRAIN];

// Forward por lote usando buffers globales
static void forward_batch(int n){
  for (int i=0;i<n;i++){
    float x = Xbuf[i] * X_SCALE;
    // oculta
    for (int j=0;j<H;j++){
      float z = W0[j]*x + b0[j];
      A1buf[i*H + j] = tanhf(z);
    }
    // salida
    float y = b1;
    for (int j=0;j<H;j++) y += W1[j]*A1buf[i*H+j];
    Yhatbuf[i] = y;
  }
}

// Un paso de SGD sobre todo el conjunto (simple)
static void sgd_step(int n, float lr){
  forward_batch(n);

  float gW0[8]={0}, gb0[8]={0};
  float gW1[8]={0}, gb1_acc=0;

  for (int i=0;i<n;i++){
    float e = Yhatbuf[i] - Ybuf[i];      // dL/dyhat
    gb1_acc += e;
    for (int j=0;j<H;j++) gW1[j] += e * A1buf[i*H+j];

    // back a oculta
    float x = Xbuf[i] * X_SCALE;
    for (int j=0;j<H;j++){
      float d = e * W1[j] * tanh_deriv_from_y(A1buf[i*H+j]);
      gb0[j] += d;
      gW0[j] += d * x;                   // in_dim=1
    }
  }

  float invn = 1.0f / (float)n;
  b1 -= lr * gb1_acc * invn;
  for (int j=0;j<H;j++){
    W1[j] -= lr * gW1[j] * invn;
    W0[j] -= lr * gW0[j] * invn;
    b0[j] -= lr * gb0[j] * invn;
  }
}

// Entrenamiento en la placa + impresión de nuevos pesos
static void tune_on_board(){
  // dataset uniforme en [-2π, 2π]
  for (int i=0;i<N_TRAIN;i++){
    float t = (float)i/(N_TRAIN-1);
    float x = -2*PI + t*(4*PI);
    Xbuf[i] = x;
    Ybuf[i] = sinf(x);
  }

  for (int e=1;e<=EPOCHS;e++){
    sgd_step(N_TRAIN, LR);
    if (e%100==0 || e==1 || e==EPOCHS){
      forward_batch(N_TRAIN);
      float mse=0;
      for (int i=0;i<N_TRAIN;i++){ float d = Yhatbuf[i]-Ybuf[i]; mse += d*d; }
      mse /= (float)N_TRAIN;
      Serial.print("epoch "); Serial.print(e);
      Serial.print("  mse="); Serial.println(mse,6);
    }
  }

  // Imprime los pesos en formato C
  Serial.println("\n--- NUEVOS PESOS (copiar y reemplazar) ---");
  Serial.print("static const float W0[8] = { ");
  for (int j=0;j<H;j++){ Serial.print(W0[j],6); Serial.print("f"); if(j<H-1) Serial.print(", "); }
  Serial.println(" };");
  Serial.print("static const float b0[8] = { ");
  for (int j=0;j<H;j++){ Serial.print(b0[j],6); Serial.print("f"); if(j<H-1) Serial.print(", "); }
  Serial.println(" };");
  Serial.print("static const float W1[8] = { ");
  for (int j=0;j<H;j++){ Serial.print(W1[j],6); Serial.print("f"); if(j<H-1) Serial.print(", "); }
  Serial.println(" };");
  Serial.print("static const float b1 = "); Serial.print(b1,6); Serial.println("f;");
  Serial.println("--- FIN ---\n");
}
#endif // TRAIN_ON_BOARD

// ===== PROGRAMA =====
void setup() {
  Serial.begin(115200);
  while(!Serial){}

#if TRAIN_ON_BOARD
  Serial.println("Entrenando MLP 1-8-1 para sin(x) en la placa...");
  tune_on_board();   // ajusta pesos y los imprime
  Serial.println("Entrenamiento listo. Ahora tabla de verificación:");
#else
  Serial.println("MLP SENO 1-8-1 (solo inferencia).");
#endif

  // Tabla de prueba
  for (float x = -2*PI; x <= 2*PI; x += PI/4) {
    float y_nn = nn_predict(x);
    float y_rf = sinf(x);
    Serial.print("x=");   Serial.print(x, 5);
    Serial.print("  nn="); Serial.print(y_nn, 6);
    Serial.print("  sin="); Serial.print(y_rf, 6);
    Serial.print("  err="); Serial.println(y_nn - y_rf, 6);
  }

  Serial.println("Escribe un angulo (radianes) y envio nn(x).");
}

void loop() {
  if (Serial.available()) {
    float x = Serial.parseFloat();
    flushLineEndings();
    Serial.println(nn_predict(x), 6);
  }
}
