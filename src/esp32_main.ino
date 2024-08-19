#include <WiFi.h>
#include <HTTPClient.h>
#include "DHT.h"
#include <time.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "Na3L";
const char* password = "@Rumah3kitA";

// Define the DHT sensor type and pin
#define DHTTYPE DHT11
#define DHT_PIN 27
DHT dht(DHT_PIN, DHTTYPE);

const char* serverName = "http://192.168.1.10:2200/data";  // Update with your Flask server IP and port

void setup() {
  Serial.begin(115200);
  delay(10);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Initialize DHT sensor
  dht.begin();

  // Initialize TimeLib and synchronize with NTP server
  configTime(7 * 3600, 0, "pool.ntp.org");

  // Wait for time to synchronize
  while (!time(nullptr)) {
    delay(1000);
    Serial.println("Waiting for time synchronization...");
  }
}

void loop() {
  delay(3000);

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  // Get current time
  String currentTime = getTimeString();

  Serial.print("Time: ");
  Serial.print(currentTime);
  Serial.print(", Temperature: ");
  Serial.print(temperature);
  Serial.print(" °C, Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");

  String jsonPayload = "{\"time\":";
  jsonPayload += currentTime;
  jsonPayload += ",\"temperature\":";
  jsonPayload += temperature;
  jsonPayload += ",\"humidity\":";
  jsonPayload += humidity;
  jsonPayload += "}";

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    Serial.print("Connecting to server: ");
    Serial.println(serverName);

    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    int retryCount = 3;  // Number of retry attempts
    int httpResponseCode = http.POST(jsonPayload);

    while (retryCount > 0 && (httpResponseCode <= 0 || httpResponseCode >= 400)) {
      httpResponseCode = http.POST(jsonPayload);

      if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.print("HTTP Response code: ");
        Serial.println(httpResponseCode);
        Serial.print("Response: ");
        Serial.println(response);
        break;
      } else {
        Serial.print("Error code: ");
        Serial.println(httpResponseCode);
        retryCount--;
        delay(1000);  // Wait before retrying
      }
    }
    http.end();
  } else {
    Serial.println("WiFi Disconnected");
  }
}

String getTimeString() {
  time_t now;
  struct tm timeinfo;
  char buffer[80];

  time(&now);
  localtime_r(&now, &timeinfo);

  strftime(buffer, sizeof(buffer), "%B %d %Y %H:%M:%S", &timeinfo);
  return String(buffer);
}
