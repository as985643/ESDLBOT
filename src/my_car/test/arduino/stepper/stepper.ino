#define dirPin 4
#define stepPin 5
#define delay_time 65
long start;
long finall;
long durring;
long pulse;

void setup() {
  Serial.begin(9600);
  // Declare pins as output:
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  // Set the spinning direction CW/CCW:
  digitalWrite(dirPin, HIGH);
  pinMode(2, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(2), blink, RISING);
}
void loop() {
  start = micros();
  for(int i=0; i<1600*2; i++){
    // These four lines result in 1 step:
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(delay_time/2);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(delay_time/2);
  }
  finall = micros();
  //Serial.println(durring);
  durring = (finall-start);
  Serial.println(pulse*1.0/durring*37500);
  pulse = 0.0;
  //delay(1000);
}
void blink() {
  pulse ++;
}
