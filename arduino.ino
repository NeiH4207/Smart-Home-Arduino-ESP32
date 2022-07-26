/*
 * This ESP32 code is created by esp32io.com
 *
 * This ESP32 code is released in the public domain
 *
 * For more detail (instruction and wiring diagram), visit https://esp32io.com/tutorials/esp32-temperature-humidity-sensor
 */

#include <DHT.h>
#include <WiFi.h>
#include <ArduinoMqttClient.h>
#include <ArduinoJson.h> 
#include <PubSubClient.h>

#define DHT_SENSOR_PIN  21 // ESP32 pin GIOP21 connected to DHT11 sensor
#define DHT_SENSOR_TYPE DHT11
#define BUTTON_PIN 16  // ESP32 pin GIOP16, which connected to button
#define LED_PIN    18  // ESP32 pin GIOP18, which connected to led
#define BUILTIN_LED 2

// The below are variables, which can be changed
int button_state = 0;   // variable for reading the button status
DHT dht_sensor(DHT_SENSOR_PIN, DHT_SENSOR_TYPE);
int LED1_status = 0;
int air_conditioner_status = 0;

/*********
  Rui Santos
  Complete project details at https://randomnerdtutorials.com  
*********/

// WiFi
const char* ssid = "TP-LINK_6EEBDA";
const char* password = "1234567890";

// MQTT Broker
const char *mqtt_broker = "broker.emqx.io";
const char *topic = "iot/status";
const char *mqtt_username = "emqx";
const char *mqtt_password = "public";
const int mqtt_port = 1883;
const char *client_id = "esp8266-client-arduino";
const int timeout = 10000;

WiFiClient espClient;
PubSubClient client(espClient);

unsigned long lastMsg = 0;
#define MSG_BUFFER_SIZE  (50)
char msg[MSG_BUFFER_SIZE];

void led_blink();

void ConnectToWifi(){
  Serial.print("Connecting to wifi");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  unsigned long startAttemptTime = millis();

  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < timeout) {
    Serial.print(".");
    delay(100);
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Fail");
  } else{
    Serial.print("Connected");
    Serial.println(WiFi.localIP());
  }
}

void callback(char *topic, byte *payload, unsigned int length) {
    Serial.print("Message arrived in topic: ");
    Serial.println(topic);
    Serial.print("Message:");
    for (int i = 0; i < length; i++) {
        Serial.print((char) payload[i]);
    }
    Serial.println("-----------------------");  
    if ((char) payload[11] == 'o' && (char) payload[12] == 'n'){
        digitalWrite(LED_PIN, HIGH); // turn on LED
        LED1_status = 1;
    }
    if ((char) payload[11] == 'o' && (char) payload[12] == 'f'){
        digitalWrite(LED_PIN, LOW); // turn on LED
        LED1_status = 0;
    }
    
}

void setup() {
  Serial.begin(115200);
  dht_sensor.begin(); // initialize the DHT sensor
    // initialize the LED pin as an output:
  pinMode(LED_PIN, OUTPUT);
  // initialize the button pin as an pull-up input:
  // the pull-up input pin will be HIGH when the button is open and LOW when the button is pressed.
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  ConnectToWifi();
  client.setServer(mqtt_broker, mqtt_port);
  client.setCallback(callback);
  while (!client.connected()) {
    Serial.println("Connecting to MQTT...");
 
    if (client.connect(client_id, mqtt_username, mqtt_password)) {
 
      Serial.println("connected");
      client.subscribe("iot/led");
 
    } else {
 
      Serial.print("failed with state ");
      Serial.print(client.state());
      delay(2000);
 
    }
  }
}

void led_blink(){
  // read the state of the button value:
  button_state = digitalRead(BUTTON_PIN);

  // control LED according to the state of button
  if (button_state == LOW){       // if button is pressedi
    digitalWrite(LED_PIN, HIGH); // turn on LED
    LED1_status = 1;
  }
  else {                          // otherwise, button is not pressing
    digitalWrite(LED_PIN, LOW);  // turn off LED
    LED1_status = 0;
  }
}

void loop() {
  // call poll() regularly to allow the library to send MQTT keep alive which
  // avoids being disconnected by the broker
//  led_blink();
  float humi  = dht_sensor.readHumidity();
  // read temperature in Celsius
  float tempC = dht_sensor.readTemperature();
  // read temperature in Fahrenheit
  float tempF = dht_sensor.readTemperature(true);

  long now = millis();
  if (now - lastMsg > 5000){
      lastMsg = now;
    // check whether the reading is successful or not
    if ( isnan(tempC) || isnan(tempF) || isnan(humi)) {
      Serial.println("Failed to read from DHT sensor!");
    } else {
      Serial.print("Humidity: ");
      Serial.print(humi);
      Serial.print("%");
  
      Serial.print("  |  ");
  
      Serial.print("Temperature: ");
      Serial.print(tempC);
      Serial.print("°C  ~  ");
      Serial.print(tempF);
      Serial.println("°F");
      // send message, the Print interface can be used to set the message contents
      StaticJsonBuffer<300> JSONbuffer;
      JsonObject& JSONencoder = JSONbuffer.createObject();
     
      JSONencoder["device"] = "ESP32";
      JSONencoder["Humidity"] = humi;
      JSONencoder["Temperature(C)"] = tempC;
      JSONencoder["Temperature(F)"] = tempF;
      JSONencoder["LED-1"] = LED1_status;
      JSONencoder["air-conditioner-status"] = air_conditioner_status;
      JSONencoder["air-conditioner-mode"] = "auto";
     
      char JSONmessageBuffer[300];
      JSONencoder.printTo(JSONmessageBuffer, sizeof(JSONmessageBuffer));
      Serial.println("Sending message to MQTT topic..");
      Serial.println(JSONmessageBuffer);
     
      if (client.publish(topic, JSONmessageBuffer) == true) {
        Serial.println("Success sending message");
      } else {
        Serial.println("Error sending message");
      }
     
      Serial.println("-------------");
      client.loop();
    }
  }

  // wait a 2 seconds between readings
  delay(2000);
}
