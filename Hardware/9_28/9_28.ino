#include <Ticker.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <WebServer.h>
#include "SPIFFS.h"

#define SCREEN_WIDTH 128 // OLED显示宽度
#define SCREEN_HEIGHT 64 // OLED显示高度
#define OLED_RESET     -1 // Reset pin
#define SCREEN_ADDRESS 0x3C // OLED地址

#define PUMP_AIN1 32
#define PUMP_AIN2 23           //ESP32的 GPIO34---GPIO39 为“仅输入引脚”
#define PUMP_PWM 19//水泵1端口

#define PUMP2_PWM 33//水泵2端口
#define PUMP3_PWM 26//水泵3端口
#define PUMP4_PWM 25//水泵4端口

#define VALVE_1 12//电磁阀1端口
#define VALVE_2 14//电磁阀2端口
#define VALVE_3 27//电磁阀3端口

#define SW_RX 4
#define SW_RY 2
#define SW_SW 15//摇杆端口

#define WATER_SENSER_1 39//水位传感器1
#define WATER_SENSER_2 35//水位传感器2
#define WATER_SENSER_3 34//水位传感器3

#define GLUCOSE_SENSER 13//反应器传感器端口

// 显示屏引脚定义
#define OLED_SDA 21
#define OLED_SCL 22



#define OPEN  1    
#define CLOSE 0
#define NONE_PIN 0
#define NONE_DIR 0
#define PART1_SPEED 180//水泵1速度
#define PART2_SPEED 110//水泵2速度
#define WATER_SENSER_STD 3400//水位传感器参考值


#define PART1_OUT_TIME 5   //装置1出液时间（包含气泡搅拌）
#define PART1_BACK_TIME 13//装置1退液时间
#define AIR_IN_TIME 12//进气搅拌时间
//#define PART2_OUT_TIME 20//装置2出液时间（包含气泡搅拌）
#define PART2_BACK_TIME 5//装置2退液时间
#define WATER_OUT_TIME 10//稀释瓶排废液时间
#define CLEAN_TIME 8//反应池清洗时间
#define READ_DATA_TIME 8//读取数据
Ticker timer;//定时中断对象
//Ticker timer2;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);


struct Joystick{
  int rx;
  int ry;
  int sw;
  int sw_last;
  int sw_long;
  int sw_b;
  int sw_c;
  int sw_timer;
  int rx_state;//正数为向上按，负数为向下按，0为没按
};
struct Senser_data{
  float data;
  float data_n;
  float data_l;
  float data_f;
  int pin;
};
struct mission_value{
  int flag;      //0为待机状态，1为正在取样
  int state;     //0为待机，1为定量环1进液，2为定量环1出液，3为定量环1退液，4为稀释瓶进水，5为稀释瓶进气(搅拌)，6为定量环2进液，7为定量环2出液，8为定量环2退液，9为读取反应池读数，10为清理装置，11为清理反应器,12为反应器装水，13为清理装置
  int count;  //取样计数器
  int gape_minute;//取样时间间隔：分钟
  int gape_hour;  //取样间隔时间：小时
  int count_target;//目标取样数
  int data;//数据存放点
  int water_in_time;//稀释时间

};
struct timer_ms{
  int year;
  int month;
  int day;
  int hour;
  int minute;
  int second;
  int ms;
  int second_total;
};

Joystick stick1;
Senser_data water_1;//液体传感器1
Senser_data water_2;//液体传感器2
Senser_data water_3;//液体传感器3
Senser_data glucose_1;//传感器
mission_value mission1;//任务结构体
timer_ms timer_1={0,0,0,9,30,0,0,0};//系统运行时钟
timer_ms timer_2={0,0,0,0,0,0,0,0};//计时时钟

float VOFA_DATA[8];//VOFA+助手，发送数据缓冲区
char vofa_tail[4]={
  0x00,0x00,0x80,0x7f
};//VOFA+助手，帧尾格式
char data_v[64];

const char* WIFI_SSID = "123456gy";
const char* WIFI_PASS = "12345678";
int wifi_flag=0;

WebServer server(3000); // ESP32 HTTP 服务端口
int localSampleCounter = 0;

// 模拟传感器读取浓度
float getConcentration() {
  return mission1.data; // 模拟浓度值
}

// 保存 JSON 数据到 SPIFFS
void saveDataToSPIFFS(String jsonStr) {
  File file = SPIFFS.open("/history.json", FILE_APPEND);
  if (!file) {
    Serial.println("无法打开文件进行写入");
    return;
  }
  file.println(jsonStr); // 每条数据换行存储
  file.close();
}

// 处理电脑端采样请求
void handleSample() {
  wifi_flag=1;
  localSampleCounter++;
  mission1.flag=1;

  // 12 步模拟采样进程，每步 200ms
  

  float concentration = getConcentration();

  // 构造 JSON
  String response = "{\"device_id\":\"ems32\","
                    "\"sample_index\":" + String(localSampleCounter) + ","
                    "\"value\":" + String(concentration) + "}";

  // 保存数据到 SPIFFS
  saveDataToSPIFFS(response);

  // 添加 CORS 头
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", response);
  wifi_flag=0;
}

// 提供历史数据接口
void handleHistory() {
  wifi_flag=1;
  File file = SPIFFS.open("/history.json", FILE_READ);
  if (!file) {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", "[]");
    return;
  }

  String allData = "[";
  while (file.available()) {
    String line = file.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      allData += line + ",";
    }
  }
  file.close();
  if (allData.endsWith(",")) allData.remove(allData.length() - 1);
  allData += "]";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", allData);
  wifi_flag=0;
}

void isr_main() {
  if(!wifi_flag){

  my_time_counter(&timer_1,50);//系统时钟计时
  my_time_counter(&timer_2,50);//计时器计时

  joystatic_read(&stick1);
  read_senser(&water_1);
  read_senser(&water_2);
  read_senser(&water_3);
  read_senser(&glucose_1);

  //joystatic_control();
  mission_judge(&mission1);
  control_test();
  
  }
}

/**********************************************************************
函数名称：my_time_counter（）
函数功能：计时器计时
参数：    *timer_ms timer
          int cycle（周期）
示例：time_counter（&timer_1,50）;
***********************************************************************/
void my_time_counter(timer_ms* timer_x,int cycle){
  timer_x->ms++;
  if(timer_x->ms>=cycle){
    timer_x->ms=0;
    timer_x->second++;
    timer_x->second_total++;
  }
  if(timer_x->second>=60){
    timer_x->second=0;
    timer_x->minute++;
  }
  if(timer_x->minute>=60){
    timer_x->minute=0;
    timer_x->hour++;
  }
  if(timer_x->hour>=60){
    timer_x->hour=0;
    timer_x->day++;
  }
}

//函数功能：清空计时器(开始计时)
void my_clear_timer_(timer_ms* timer_x){

  timer_x->ms=0;
  timer_x->second=0;
  timer_x->minute=0;
  timer_x->hour=0;
  timer_x->day=0;
  timer_x->second_total=0;
  //timer_x->month=0;
}

//初始化
void setup() {
  pinMode(SW_RX,INPUT);
  pinMode(SW_RY,INPUT);
  pinMode(SW_SW,INPUT_PULLUP);//遥感初始化

  pinMode(WATER_SENSER_1,INPUT);//初始化水位传感器端口
  pinMode(WATER_SENSER_2,INPUT);//初始化水位传感器端口
  pinMode(WATER_SENSER_3,INPUT);//初始化水位传感器端口
  pinMode(GLUCOSE_SENSER,INPUT);//初始化水位传感器端口

  pinMode(PUMP_AIN1,OUTPUT);
  pinMode(PUMP_AIN2,OUTPUT);
  pinMode(PUMP_PWM,OUTPUT);
  pinMode(PUMP2_PWM,OUTPUT);
  pinMode(PUMP3_PWM,OUTPUT);
  pinMode(PUMP4_PWM,OUTPUT);

  pinMode(VALVE_1,OUTPUT);
  pinMode(VALVE_2,OUTPUT);
  pinMode(VALVE_3,OUTPUT);




  senser_data_init(WATER_SENSER_1,&water_1);//初始化压力传感器数据
  senser_data_init(WATER_SENSER_2,&water_2);//初始化压力传感器数据
  senser_data_init(WATER_SENSER_3,&water_3);//初始化压力传感器数据
  senser_data_init(GLUCOSE_SENSER,&glucose_1);//初始化水位传感器数据
  joystatic_setup(&stick1);//摇杆数据初始化
  mission_data_init(&mission1);

  timer.attach_ms(20,isr_main);//定时中断使能
  wifi_flag=1;
  //timer.attach_ms(1,time_counter);//定时中断使能
  Serial.begin(115200);//串口初始化
  Wire.begin(OLED_SDA, OLED_SCL);
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // 卡住不继续执行
  }
  
  // 显示系统启动信息
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("System Starting...");
  display.display();
  delay(2000);

  // 初始化 SPIFFS
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Initializing SPIFFS");
  display.display();
  
  if (!SPIFFS.begin(true)) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("SPIFFS Init Failed!");
    display.setCursor(0, 16);
    display.println("Using RAM storage");
    display.display();
    delay(3000);
  } else {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("SPIFFS Ready");
    display.display();
    delay(500);
  }

  // 连接 WiFi
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Connecting to WiFi:");
  display.setCursor(0, 16);
  display.print("SSID: ");
  display.println(WIFI_SSID);
  display.setCursor(0, 32);
  display.print("Status: ");
  display.display();
  
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int dotCount = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    
    // 更新连接状态显示
    display.fillRect(60, 32, 68, 8, BLACK); // 清除状态区域
    display.setCursor(60, 32);
    for (int i = 0; i < dotCount; i++) {
      display.print(".");
    }
    display.display();
    
    dotCount = (dotCount + 1) % 4; // 0-3个点循环
  }

  // WiFi 连接成功
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("WiFi Connected!");
  display.setCursor(0, 16);
  display.print("IP: ");
  display.println(WiFi.localIP());
  display.display();
  delay(3500);

  // 启动 Web 服务器
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Starting Web Server");
  display.setCursor(0, 16);
  display.println("Port: 3000");
  display.display();
  
  server.on("/sample", handleSample);
  server.on("/history", handleHistory);
  server.begin();
  
  delay(2500);
  
  // 最终准备完成显示
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("System Ready");
  display.setCursor(0, 16);
  display.println("IP: " + WiFi.localIP().toString());
  display.setCursor(0, 32);
  display.println("Port: 3000");
  display.setCursor(0, 48);
  display.println("Sampling Count: 0");
  display.display();
  delay(3000);
  
  // 清屏准备进入主循环
  display.clearDisplay();
  display.display();
  wifi_flag=0;
}

void updateOLEDDisplay() {
  display.clearDisplay();
  
  // 显示时间
  display.setCursor(0, 0);
  display.print("Time: ");
  display.print(timer_1.hour);
  display.print(":");
  if(timer_1.minute < 10) display.print("0");
  display.print(timer_1.minute);
  display.print(":");
  if(timer_1.second < 10) display.print("0");
  display.print(timer_1.second);
  
  // 显示状态
  display.setCursor(0, 16);
  display.print("State: ");
  if(mission1.flag) {
    display.print("Working");
    display.setCursor(0, 32);
    display.print("Progress: ");
    display.print(mission1.state);
    display.print("/13");
  } else {
    if(stick1.sw_b==3){
      display.print("Waitting");
    }
    else if(stick1.sw_b==0){
      display.print("Cleaning");
    }
    else if(stick1.sw_b==1){
      display.print("Debugging1");
    }
    else{
      display.print("Debugging2");
    }
    display.setCursor(0, 32);
    display.print("Sampling times: ");
    display.print(mission1.count);
  }
  
  // 显示传感器数据
  //display.setCursor(0, 48);
  /*for(int i=0;i<(0.005*stick1.rx);i++){
 
    display.print("■"); 
  }*/
  display.fillRect(0, 48, 0.03125*stick1.rx, 10, WHITE);
  
  
  
  display.display();
}

void loop() {

  //delay(5000);
 VOFA_DATA[0]=float(stick1.rx);
 //VOFA_DATA[1]=float(stick1.ry);
 VOFA_DATA[1]=float(stick1.sw_b);
 VOFA_DATA[2]=water_1.data_f;
 VOFA_DATA[3]=water_2.data_f;
 VOFA_DATA[4]=water_3.data_f;
 VOFA_DATA[5]=mission1.state;
 VOFA_DATA[6]=mission1.flag;
 VOFA_DATA[7]=glucose_1.data_f;
 vofa_send_data(VOFA_DATA,8);

 updateOLEDDisplay();
 server.handleClient(); // 处理客户端请求
 delay(1000);
 
}

/**********************************************************************
函数名称：pump_control（）
函数功能：控制水泵运转
参数：    ain1  ain2  pwm
          dir ：方向 （1为正转）
          speed ：速度（0-255）
示例：pump_control(PUMP_AIN1,PUMP_AIN2,PUMP_PWM,1,200);
***********************************************************************/
void pump_control(int ain1,int ain2,int pwm,int dir,int speed){

 digitalWrite(ain1,dir);  
 digitalWrite(ain2,!(dir));
 analogWrite(pwm,speed);

}

/**********************************************************************
函数名称：joystatic_read（）
函数功能：读取遥感数值
参数：    Joystick* stick
示例：joystatic_read(&stick1);
***********************************************************************/
void joystatic_read(Joystick* stick){
  stick->rx=analogRead(SW_RX);
  stick->ry=analogRead(SW_RY);
  stick->sw_last=stick->sw;
  stick->sw=!digitalRead(SW_SW);
  if(stick->sw==1){
    stick->sw_timer++;
  }
  if(stick->sw==0&&stick->sw_last==1){
    if(stick->sw_timer<=20){
      stick->sw_b++;
      stick->sw_b%=4;
      stick->sw_c++;
      //mission1.state=0;
  }
    else{
      stick->sw_long=!stick->sw_long;
    }
  stick->sw_timer=0;
  }
  if(stick->rx>4000){
    stick->rx_state++;
  }
  else if(stick->rx<100){
    stick->rx_state--;
  }
  else {
    stick->rx_state=0;
  }
}

//摇杆数据初始化
void joystatic_setup(Joystick* stick){
  stick->sw=0;
  stick->sw_last=0;
  stick->sw_b=3;
  stick->sw_c=0;
  stick->sw_long=0;
  stick->sw_timer=0;
}
//任务数据初始化
void mission_data_init(mission_value* my_mission){
  my_mission->flag=0;
  my_mission->state=0;
  my_mission->count=0;
  my_mission->gape_minute=0;
  my_mission->gape_hour=0;
  my_mission->count_target=0;
  my_mission->water_in_time=14;
}

/**********************************************************************
函数名称：vofa_send_data（）
函数功能：发送数值至VOFA+绘图
参数：    float* data ： 数据地址
          int num   ：  数据量
示例： vofa_send_data(VOFA_DATA,4);
***********************************************************************/
void vofa_send_data(float* data,int num){

  for(int i=0;i<num;i++){
    Serial.print(data[i]);
    if(i==(num-1)){
      Serial.println();
    }
    else{
      Serial.print(",");
    }
  }

}



/**********************************************************************
函数名称：valve_control（）
函数功能：电磁阀控制
参数：    int flag ： 开关标志位
          int pin   ：  控制端口
示例： valve_control(0,VALVE_1);
***********************************************************************/
void valve_control(int flag,int pin){
  if(flag){
    analogWrite(pin,140);
  }
  else{
    analogWrite(pin,0);
  }
}



/**********************************************************************
函数名称：read_senser（）
函数功能：读取传感器数据
参数：    Senser_data* senser
示例： read_senser(&pressure_1);
***********************************************************************/
float read_senser(Senser_data* senser){
  senser->data_l=senser->data_n;
  senser->data_n=senser->data;
  senser->data=analogRead(senser->pin);
  senser->data_f=0.7*senser->data_f+0.3*senser->data;
  return senser->data;
}


/**********************************************************************
函数名称：senser_data_init（）
函数功能：传感器数据初始化
参数：    int pin ： 端口号
          Senser_data* senser
示例： senser_data_init(PRESSURE_SENSER,&pressure_1);
***********************************************************************/
void senser_data_init(int pin,Senser_data* senser){
  senser->data_l=0;
  senser->data_n=0;
  senser->data=0;
  senser->data_f=0;
  senser->pin=pin;
  pinMode(pin,INPUT);
}


//摇杆控制，调试用
void joystatic_control(void){
  if(stick1.sw_b==0){
    if(stick1.rx_state>5){
      Part1_back(PART1_SPEED);
    }
    else if(stick1.rx_state<-5){
      Water_out(PART1_SPEED);
    }
    else{
      Close_Part();
      Close_Water();
    }
  }
  if(stick1.sw_b==1){
    if(stick1.rx_state>5){
      Part2_in(PART1_SPEED);
    }
    else if(stick1.rx_state<-5){
      Part2_back(PART1_SPEED);
    }
    else{
      Close_Part();
      Close_Water();
    }
  }
  else if(stick1.sw_b==2){
    if(stick1.rx_state>5){
  
      Part2_out(PART2_SPEED);
    
    }
    else if(stick1.rx_state<-5){
      Water_in(PART1_SPEED);
      
    }
    else{
     Close_Part();
     Close_Water();
   }
  }

  }



//控制装置测试
void control_test(void){
   if(mission1.state==0){
        Close_Part();
        Close_Water();
        joystatic_control();
   }
   else if(mission1.state==1){
      Part1_in(PART1_SPEED);
      Close_Water();
   }
   else if(mission1.state==2){
      Part1_out(PART1_SPEED);
      Close_Water();
   }
   else if(mission1.state==3){
      Part1_back(PART1_SPEED);
      Close_Water();
   }
   else if(mission1.state==4){
      Water_in(PART1_SPEED);
      Close_Part();
   }
   else if(mission1.state==5){
      Part1_out(250);
      Close_Water();
   }
   else if(mission1.state==6){
      Part2_in(PART1_SPEED);
      Close_Water();
   }
   else if(mission1.state==7){
      Part2_out(PART2_SPEED);
      Close_Water();
   }
   else if(mission1.state==8){
      Part2_back(PART1_SPEED);
      Close_Water();
   }
   else if(mission1.state==9){
      Close_Water();
      Close_Part();
      mission1.data=glucose_1.data_f;
   }
   else if(mission1.state==10){
      Water_out(PART1_SPEED);
      Close_Part();
   }
   else if(mission1.state==11){
      Buffer_in(PART2_SPEED);
      Close_Part();
   }
   else if(mission1.state==12){
      Water_in(PART1_SPEED);
      Close_Part();
   }
   else if(mission1.state==13){
      Water_out(PART1_SPEED);
      Close_Part();
   }
}

//任务变量判断
void mission_judge(mission_value* mission){
  
  if(stick1.sw_b==3){
    if(stick1.rx_state>5){
  
      mission->flag=1;
    
    }
    else if(stick1.rx_state<-5){
      mission1.flag=0;
      
    }
  }
  if(mission->flag){
    if(mission->state==0){
      mission->state=1;
      
    }
    if(mission->state==1&&water_1.data<3980)//如果定量管装满水
    {
      mission->state=2;
      my_clear_timer_(&timer_2);//开始计时
    }
    if(mission->state==2&&timer_2.second_total>=PART1_OUT_TIME)//出液一定时间后
    {
      mission->state=3;
      my_clear_timer_(&timer_2);//开始计时
    }
    if(mission->state==3&&timer_2.second_total>=PART1_BACK_TIME)//退液一定时间后
    {
      mission->state=4;
      my_clear_timer_(&timer_2);//开始计时
    }
    if(mission->state==4&&timer_2.second_total>=mission->water_in_time)//进水一定时间后
    {
      mission->state=5;
      my_clear_timer_(&timer_2);//开始计时
    }
    if(mission->state==5&&timer_2.second_total>=AIR_IN_TIME)//进气搅拌一定时间后
    {
      mission->state=6;
    }
    if(mission->state==6&&water_2.data<4000)//如果定量管2装满水
    {
      mission->state=11;
      my_clear_timer_(&timer_2);//开始计时
    }
    if(mission->state==11&&timer_2.second_total>=CLEAN_TIME)//如果反应器清理了一段时间
    {
      mission->state=7;
      my_clear_timer_(&timer_2);//开始计时
    }
    if(mission->state==7&&water_3.data>4000&&timer_2.second_total>=4)//如果定量管2进完样液
    {
      mission->state=8;
      my_clear_timer_(&timer_2);//开始计时
    }
    if(mission->state==8&&timer_2.second_total>=PART2_BACK_TIME)//如果定量管2退液一段时间
    {
      mission->state=9;
      my_clear_timer_(&timer_2);//开始计时
    }
    if(mission->state==9&&timer_2.second_total>=READ_DATA_TIME)//如果反应一段时间
    {
      mission->state=10;
      my_clear_timer_(&timer_2);//开始计时
    }
    if(mission->state==10&&timer_2.second_total>=WATER_OUT_TIME)//如果清理装置一段时间
    {
      mission->state=12;
      my_clear_timer_(&timer_2);//开始计时
    }
    if(mission->state==12&&timer_2.second_total>=(mission->water_in_time+2))//如果进水一段时间
    {
      mission->state=13;
      my_clear_timer_(&timer_2);//开始计时
    }
    if(mission->state==13&&timer_2.second_total>=WATER_OUT_TIME)//如果清理装置一段时间
    {
      mission->state=0;
      mission->flag=0;
      mission->count++;
    }

  }
  else{
      mission->state=0;
  }
  

}

//任务决策
/*void mission_dission(mission_value* mission){

}*/



//以下部分为动作模组


//********************定量环1进液
void Part1_in(int speed){
  valve_control(0,VALVE_3);
  valve_control(0,VALVE_2);
  valve_control(0,VALVE_1);
  pump_control(PUMP_AIN1,PUMP_AIN2,PUMP_PWM,1,speed);
}

//********************定量环1出液
void Part1_out(int speed){
  valve_control(0,VALVE_3);
  valve_control(1,VALVE_2);
  valve_control(0,VALVE_1);
  pump_control(PUMP_AIN1,PUMP_AIN2,PUMP_PWM,0,speed);
}

//********************定量环1退液
void Part1_back(int speed){
  valve_control(0,VALVE_3);
  valve_control(0,VALVE_2);
  valve_control(0,VALVE_1);
  pump_control(PUMP_AIN1,PUMP_AIN2,PUMP_PWM,0,speed);
}

//********************定量环2进液
void Part2_in(int speed){
  valve_control(0,VALVE_3);
  valve_control(0,VALVE_2);
  valve_control(1,VALVE_1);
  pump_control(PUMP_AIN1,PUMP_AIN2,PUMP_PWM,1,speed);
}

//********************定量环2出液
void Part2_out(int speed){
  valve_control(1,VALVE_3);
  valve_control(0,VALVE_2);
  valve_control(1,VALVE_1);
  pump_control(PUMP_AIN1,PUMP_AIN2,PUMP_PWM,0,speed);
}

//********************定量环2退液
void Part2_back(int speed){
  valve_control(0,VALVE_3);
  valve_control(0,VALVE_2);
  valve_control(1,VALVE_1);
  pump_control(PUMP_AIN1,PUMP_AIN2,PUMP_PWM,0,speed);
}

//********************关闭定量环所有功能
void Close_Part(){
  valve_control(0,VALVE_3);
  valve_control(0,VALVE_2);
  valve_control(0,VALVE_1);
  pump_control(PUMP_AIN1,PUMP_AIN2,PUMP_PWM,0,0);
}

//********************加水稀释
void Water_in(int speed){
  pump_control(NONE_PIN,NONE_PIN,PUMP2_PWM,NONE_DIR,speed);
  pump_control(NONE_PIN,NONE_PIN,PUMP4_PWM,NONE_DIR,0);
  pump_control(NONE_PIN,NONE_PIN,PUMP3_PWM,NONE_DIR,0);
}
//********************加入缓冲液
void Buffer_in(int speed){
  pump_control(NONE_PIN,NONE_PIN,PUMP3_PWM,NONE_DIR,speed);
  pump_control(NONE_PIN,NONE_PIN,PUMP4_PWM,NONE_DIR,0);
  pump_control(NONE_PIN,NONE_PIN,PUMP2_PWM,NONE_DIR,0);
}
//********************排出水
void Water_out(int speed){
  pump_control(NONE_PIN,NONE_PIN,PUMP4_PWM,NONE_DIR,speed);
  pump_control(NONE_PIN,NONE_PIN,PUMP2_PWM,NONE_DIR,0);
  pump_control(NONE_PIN,NONE_PIN,PUMP3_PWM,NONE_DIR,0);
}

//********************关闭部分水泵
void Close_Water(){
  pump_control(NONE_PIN,NONE_PIN,PUMP4_PWM,NONE_DIR,0);
  pump_control(NONE_PIN,NONE_PIN,PUMP3_PWM,NONE_DIR,0);
  pump_control(NONE_PIN,NONE_PIN,PUMP2_PWM,NONE_DIR,0);
}
