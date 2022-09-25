#define LED_PIN 13

#include <vector>
//#include "altro.h"
//#include "mylib.h"
//#include "ArduinoEigen.h"
//#include "EmbeddedMPC.h"
#include "altro.h"
//#include "utils/utils.hpp"
//#include "Eigen/Dense"

void setup() {
  pinMode(LED_PIN, OUTPUT);
//  int c = MyLibAdd(1, 3);
  int c = AltroSum(2, 1);
  (void)c;
}

void loop() {
  digitalWrite(LED_PIN, HIGH);
  delay(1000);
  digitalWrite(LED_PIN, LOW);
  delay(1000);
}