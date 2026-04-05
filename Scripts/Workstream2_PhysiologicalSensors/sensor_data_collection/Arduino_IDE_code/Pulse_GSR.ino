const int pulsePin = A1;
const int GSR = A0;
const int blinkPin = 13;
const int fadePin = 5;

int fadeRate = 0;
int sensorValue = 0;
int gsr_average = 0;

volatile int BPM;
volatile int Signal;
volatile int IBI = 600;
volatile boolean Pulse = false;
volatile boolean QS = false;

unsigned long lastPrintTime = 0;
const unsigned long printInterval = 200;

void setup() {
  pinMode(blinkPin, OUTPUT);
  pinMode(fadePin, OUTPUT);
  Serial.begin(9600);
  interruptSetup();
}

void loop() {
  long sum = 0;
  for (int i = 0; i < 10; i++) {
    sensorValue = analogRead(GSR);
    sum += sensorValue;
    delay(5);
  }
  gsr_average = sum / 10;

  if (QS == true) {
    fadeRate = 255;
    QS = false;
  }

  unsigned long currentTime = millis();
  if (currentTime - lastPrintTime >= printInterval) {
    lastPrintTime = currentTime;

    Serial.print("BPM:");
    Serial.print(BPM);
    Serial.print(",GSR:");
    Serial.println(gsr_average);
  }

  ledFadeToBeat();
}

void ledFadeToBeat() {
  fadeRate -= 15;
  fadeRate = constrain(fadeRate, 0, 255);
  analogWrite(fadePin, fadeRate);
}