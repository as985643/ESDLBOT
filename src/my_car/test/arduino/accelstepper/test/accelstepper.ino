#include <ros.h>
#include <geometry_msgs/Twist.h>

#include <AccelStepper.h>

// The Left Stepper pins
#define STEPPER1_DIR_PIN 33
#define STEPPER1_STEP_PIN 25
// The Right stepper pins
#define STEPPER2_DIR_PIN 19
#define STEPPER2_STEP_PIN 18

float wheel_diameter = 0.067;
float car_width = 0.243;
int Pulse_Per_Rev = 800;

ros::NodeHandle  nh;

TaskHandle_t Task1;   //宣告任務變數Task1
TaskHandle_t Task2;   //宣告任務變數Task2

// Define some steppers and the pins the will use
AccelStepper stepper1(AccelStepper::DRIVER, STEPPER1_STEP_PIN, STEPPER1_DIR_PIN);
AccelStepper stepper2(AccelStepper::DRIVER, STEPPER2_STEP_PIN, STEPPER2_DIR_PIN);

char txt[6];
char msg[50];
void messageCb( const geometry_msgs::Twist& vel_msg){
    float Vx = vel_msg.linear.x;
    float diff = vel_msg.angular.z * car_width / 2;

    float LeftSpeed = (Vx - diff) / (PI * wheel_diameter) * Pulse_Per_Rev;
    float RightSpeed = (Vx + diff) / (PI * wheel_diameter) * Pulse_Per_Rev;
    
    if ((LeftSpeed + RightSpeed) > 4000) nh.loginfo("Pulse per second over 4000 !");
    else{
        stepper1.setSpeed(LeftSpeed);    
        stepper2.setSpeed(RightSpeed);
        /*dtostrf(RightSpeed, 3, 3, txt);
        sprintf(msg, "Right_rpm = %s", txt);
        nh.loginfo(msg);*/
    }
}

ros::Subscriber<geometry_msgs::Twist> sub("cmd_vel", messageCb );

void setup(){      
    nh.initNode();

    //** Get Params **//
    //nh.getParam("wheel_diameter", &wheel_diameter);  
    //nh.getParam("car_width", &car_width);
    //nh.getParam("Pulse_Per_Rev", &Pulse_Per_Rev);

    nh.subscribe(sub);

    // Set mpu6050
    mpu_setup();
    
    // Set max speed is 300 rpm
    stepper1.setMaxSpeed(4000.0);
    stepper2.setMaxSpeed(4000.0);

    //建立Task1任務並指定在核心0中執行
    xTaskCreatePinnedToCore(
                    mpu_loop,   /* 任務函數 */
                    "Task1",     /* 任務名稱 */
                    10000,       /* 任務推疊大小 */
                    NULL,        /* 任務參數 */
                    1,           /* 任務優先權(0是最低優先權 */
                    NULL,      /* 欲追蹤處理的任務名稱 (可使用 &Task1 或 NULL) */
                    0);          /* 指定此任務的執行核心(0或1) */

    //建立Task2任務並指定在核心0中執行
    /*xTaskCreatePinnedToCore(
                    motor_run,   // 任務函數 
                    "Task2",     // 任務名稱 
                    10000,       // 任務推疊大小 
                    NULL,        // 任務參數 
                    1,           // 任務優先權(0是最低優先權
                    NULL,        // 欲追蹤處理的任務名稱 (可使用 &Task1 或 NULL) 
                    1);          // 指定此任務的執行核心(0或1) */
 

}


void motor_run(void * pvParameters){
    for(;;){
        stepper1.runSpeed();
        stepper2.runSpeed();
    }
}
void loop(){
    //stepper1.runSpeed();
    //stepper2.runSpeed();
    //mpu_loop();
    nh.spinOnce();
}
