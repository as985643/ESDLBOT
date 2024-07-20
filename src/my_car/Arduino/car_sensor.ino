#include <Wire.h>
#include "SensirionI2cScd30.h"
#include "SHT85.h"
#include "RTC.h"
#include <WiFi.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <time.h>

// Replace with your network credentials
const char* ssid     = "Chou's ESDLab";
const char* password = "0955521135";

WiFiUDP ntpUDP;
// Initialize NTP client for time synchronization
NTPClient timeClient(ntpUDP, "pool.ntp.org", 60 * 60 * 8); // Add timezone offset (in seconds)

SensirionI2cScd30 scd30;
SHT85 sht85;
RTCTime currentTime;

String urlencode(String str)
{
    String encodedString="";
    char c;
    char code0;
    char code1;
    char code2;
    for (int i =0; i < str.length(); i++){
      c=str.charAt(i);
      if (c == ' '){
        encodedString+= '+';
      } else if (isalnum(c)){
        encodedString+=c;
      } else{
        code1=(c & 0xf)+'0';
        if ((c & 0xf) >9){
            code1=(c & 0xf) - 10 + 'A';
        }
        c=(c>>4)&0xf;
        code0=c+'0';
        if (c > 9){
            code0=c - 10 + 'A';
        }
        code2='\0';
        encodedString+='%';
        encodedString+=code0;
        encodedString+=code1;
        //encodedString+=code2;
      }
      yield();
    }
    return encodedString;
}


// Initialize record_no to 1
int record_no = 1;

void setup() {
  Wire.begin();
  Serial.begin(115200);
  scd30.begin(Wire, 0x61); // Initialize the SCD30 sensor
  sht85.begin(); // Initialize the SHT85 sensor
  RTC.begin(); // Initialize the RTC

  // Connect to WiFi and initialize NTP client for time synchronization
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  timeClient.begin();

  // Check and print the firmware version
  uint8_t major, minor;
  scd30.readFirmwareVersion(major, minor);
  Serial.print("SCD30 Firmware version: ");
  Serial.print(major);
  Serial.print(".");
  Serial.println(minor);

  // Start periodic measurement with an ambient pressure compensation of 0 mBar
  scd30.startPeriodicMeasurement(0);
}

void loop() {
  delay(30000); // Wait for 30 seconds

  // Update the NTP client and get the current time
  timeClient.update();
  unsigned long epochTime = timeClient.getEpochTime();

  // Convert epoch time to date and time
  time_t rawtime = (time_t)epochTime;
  struct tm * ptm;
  ptm = gmtime ( &rawtime );
  // ptm->tm_hour+=8; // Your timezone (Taiwan is GMT+8)
  if (ptm->tm_hour>=24){ // the day roll-over
     ptm->tm_hour-=24;
     ptm->tm_mday+=1;
  }
  mktime(ptm); // Call mktime: time.h should normalize ptm struct

  char date[20], t[10];
  sprintf(date, "%04d-%02d-%02d", ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday);
  sprintf(t, "%02d:%02d:%02d", ptm->tm_hour, ptm->tm_min, ptm->tm_sec);

  // Combine date and time
  String currentDateTime = String(date) + " " + String(t);


  // Check if new measurement data is available
  uint16_t interval;
  scd30.getMeasurementInterval(interval);
  if (interval > 0) {
    float co2, temperature, humidity;

    // Read the measurements
    scd30.readMeasurementData(co2, temperature, humidity);

    // Read and print the SHT85 measurements
    if (sht85.read()) {
      float sht85_temperature = sht85.getTemperature();
      float sht85_humidity = sht85.getHumidity();
      Serial.print("Temperature: ");
      Serial.print(temperature);
      Serial.print(" C, Humidity: ");
      Serial.print(humidity);
      Serial.print(" %,");
    }

    // Print the CO2 measurement
    Serial.print(" CO2: ");
    Serial.print(co2);
    Serial.print(" ppm,");

    // Print the current time
    Serial.print(" time: ");
    Serial.println(currentDateTime);

    // Send the measurements to your server
    if ((WiFi.status() == WL_CONNECTED)) {
      WiFiClient client;
      if (client.connect("192.168.1.180", 8000)) {
        String url = "/sql_upload?record_no=" + String(record_no) + "&area=A1" + "&temperature=" + String(temperature) + 
                      "&humidity=" + String(humidity) + "&co2=" + String(co2) + "&time=" + urlencode(currentDateTime);
        client.print(String("GET ") + url + " HTTP/1.1\r\n" +
                     "Host: " + "192.168.1.180" + "\r\n" + 
                     "Connection: close\r\n\r\n");
      }
    }
    else {
      Serial.println("WiFi Disconnected");
    }

    // Increment the record number
    record_no++;
  }
}
