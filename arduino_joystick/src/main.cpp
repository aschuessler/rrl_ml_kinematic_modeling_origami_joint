#include "main.h"
// Timing control
Timer Timer_01, Timer_02, Timer_03;

// https://stackoverflow.com/questions/9072320/split-string-into-string-array
String getValue(String data, char separator, int index)
{
  int found = 0;
  int strIndex[] = {0, -1};
  int maxIndex = data.length()-1;

  for(int i=0; i<=maxIndex && found<=index; i++){
    if(data.charAt(i)==separator || i==maxIndex){
        found++;
        strIndex[0] = strIndex[1]+1;
        strIndex[1] = (i == maxIndex) ? i+1 : i;
    }
  }
  return found>index ? data.substring(strIndex[0], strIndex[1]) : "";
}

void setup() {
  // reserve 200 bytes for the inputString:
  inputString.reserve(200);
  
  pinMode(potPin, INPUT);

  DEBUG_SERIAL.begin(baudrate_debug);
  dxl.begin(baudrate_dxl);
  
  // Set Port Protocol Version. This has to match with DYNAMIXEL protocol version.
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  // --- SYNC READ POSITION!!!
  // Fill the members of structure to syncRead using external user packet buffer
  sr_infos.packet.p_buf = user_pkt_buf;
  sr_infos.packet.buf_capacity = user_pkt_buf_cap;
  sr_infos.packet.is_completed = false;
  sr_infos.addr = SR_START_ADDR;
  sr_infos.addr_length = SR_ADDR_LEN;
  sr_infos.p_xels = info_xels_sr;
  sr_infos.xel_count = 0;

  // Prepare the SyncRead structure
  for(int i = 0; i < DXL_ID_CNT; i++) {
    info_xels_sr[i].id = DXL_ID[i];
    info_xels_sr[i].p_recv_buf = (uint8_t*)&sr_data[i];
    sr_infos.xel_count++;
  }
  sr_infos.is_info_changed = true;


  // --- SYNC WRITE POSITION!!!
  sw_infos.packet.p_buf = nullptr;
  sw_infos.packet.is_completed = false;
  sw_infos.addr = SW_START_ADDR;
  sw_infos.addr_length = SW_ADDR_LEN;
  sw_infos.p_xels = info_xels_sw;
  sw_infos.xel_count = 0;

  for(int i = 0; i < DXL_ID_CNT; i++) {
    info_xels_sw[i].id = DXL_ID[i];
    info_xels_sw[i].p_data = (uint8_t*)&sw_data[i].goal_position;
    sw_infos.xel_count++;
  }
  sw_infos.is_info_changed = true;

  int max_position_limit = 2500, min_position_limit = 700; // [0.088 deg]
  int max_motor_speed = 100; // for trajectory following 25

  for (int i=0; i<DXL_ID_CNT; i++){
    // Turn off torque when configuring items in EEPROM area
    dxl.torqueOff(DXL_ID[i]);

    // Set the mode to current based position control
    dxl.setOperatingMode(DXL_ID[i], OP_CURRENT_BASED_POSITION);

    dxl.writeControlTableItem(MAX_POSITION_LIMIT, DXL_ID[i], max_position_limit); // [0.088 deg]
    dxl.writeControlTableItem(MIN_POSITION_LIMIT, DXL_ID[i], min_position_limit); 

    // Profile Velocity	RW	0~32,767	0.229 [rev/min] ~ 0.024 rad/s
    dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID[i], max_motor_speed);
  }
 
  for (int i=0; i<DXL_ID_CNT; i++){
    // Turn on torque
    dxl.torqueOn(DXL_ID[i]);
  }

  // Set Position PID Gains  
  //double pidPGain = 400, pidDGain = 200, pidIGain = 100; // for slower motion
  double pidPGain = 8000, pidDGain = 200, pidIGain = 2000; // for faster motion
  
  for (int i=0; i<DXL_ID_CNT; i++){
    dxl.setGoalCurrent(DXL_ID[i], goalCurrent);
    dxl.writeControlTableItem(POSITION_P_GAIN, DXL_ID[i], pidPGain);
    dxl.writeControlTableItem(POSITION_D_GAIN, DXL_ID[i], pidDGain);
    dxl.writeControlTableItem(POSITION_I_GAIN, DXL_ID[i], pidIGain);
  }
  
  delay(2000);
  startTheExperiment = true;

  Timer_01.initialize(postionContolLoop);
  Timer_02.initialize(printLoop);
  Timer_03.initialize(changePositionLoop);
  
}

void loop() {
  if(Timer_01.Tick()){ 

    if(Timer_03.Tick()){
    // assign setAngles for trajectory following
      if (trajType == "roll"){
        if (currentTrajStep < TrajSteps){
          setLegAngles[0] = PI * rollTrajectoryJoint1[currentTrajStep] / 180;
          setLegAngles[1] = PI * rollTrajectoryJoint2[currentTrajStep] / 180;
          setLegAngles[2] = PI * rollTrajectoryJoint3[currentTrajStep] / 180;
          setLegAngles[3] = PI * rollTrajectoryJoint4[currentTrajStep] / 180;}
        else{
          setLegAngles[0] = PI * rollTrajectoryJoint1[TrajSteps-1] / 180;
          setLegAngles[1] = PI * rollTrajectoryJoint2[TrajSteps-1] / 180;
          setLegAngles[2] = PI * rollTrajectoryJoint3[TrajSteps-1] / 180;
          setLegAngles[3] = PI * rollTrajectoryJoint4[TrajSteps-1] / 180;}
        
      }
      else if (trajType == "pitch"){
        if (currentTrajStep < TrajSteps){
          setLegAngles[0] = PI * pitchTrajectoryJoint1[currentTrajStep] / 180;
          setLegAngles[1] = PI * pitchTrajectoryJoint2[currentTrajStep] / 180;
          setLegAngles[2] = PI * pitchTrajectoryJoint3[currentTrajStep] / 180;
          setLegAngles[3] = PI * pitchTrajectoryJoint4[currentTrajStep] / 180;}
        else{
          setLegAngles[0] = PI * pitchTrajectoryJoint1[TrajSteps-1] / 180;
          setLegAngles[1] = PI * pitchTrajectoryJoint2[TrajSteps-1] / 180;
          setLegAngles[2] = PI * pitchTrajectoryJoint3[TrajSteps-1] / 180;
          setLegAngles[3] = PI * pitchTrajectoryJoint4[TrajSteps-1] / 180;}
      }
      else if(trajType == "yaw"){
        if (currentTrajStep < TrajSteps){
          setLegAngles[0] = PI * yawTrajectoryJoint1[currentTrajStep] / 180;
          setLegAngles[1] = PI * yawTrajectoryJoint2[currentTrajStep] / 180;
          setLegAngles[2] = PI * yawTrajectoryJoint3[currentTrajStep] / 180;
          setLegAngles[3] = PI * yawTrajectoryJoint4[currentTrajStep] / 180;}
        else{
          setLegAngles[0] = PI * yawTrajectoryJoint1[TrajSteps-1] / 180;
          setLegAngles[1] = PI * yawTrajectoryJoint2[TrajSteps-1] / 180;
          setLegAngles[2] = PI * yawTrajectoryJoint3[TrajSteps-1] / 180;
          setLegAngles[3] = PI * yawTrajectoryJoint4[TrajSteps-1] / 180;}
      }
      else if(trajType == "circle"){
        if (currentTrajStep < TrajSteps){
          setLegAngles[0] = PI * circleTrajectoryJoint1[currentTrajStep] / 180;
          setLegAngles[1] = PI * circleTrajectoryJoint2[currentTrajStep] / 180;
          setLegAngles[2] = PI * circleTrajectoryJoint3[currentTrajStep] / 180;
          setLegAngles[3] = PI * circleTrajectoryJoint4[currentTrajStep] / 180;}
        else{
          setLegAngles[0] = PI * circleTrajectoryJoint1[TrajSteps-1] / 180;
          setLegAngles[1] = PI * circleTrajectoryJoint2[TrajSteps-1] / 180;
          setLegAngles[2] = PI * circleTrajectoryJoint3[TrajSteps-1] / 180;
          setLegAngles[3] = PI * circleTrajectoryJoint4[TrajSteps-1] / 180;}
      }
      else if(trajType == "serial"){
        while (DEBUG_SERIAL.available()>0) {
          inputString = DEBUG_SERIAL.readStringUntil('\n');
          stringComplete = true;
          
          for (int i=0; i<DXL_ID_CNT; i++){
            dxl.setGoalCurrent(DXL_ID[i], getValue(inputString, ' ', 4).toInt());
          }
        }

        if(stringComplete){
          for(int p = 0; p < 4 ; p++){ setLegAngles[p] = getValue(inputString, ' ', p).toFloat(); 
          }
          inputString = ""; //Reset string
          stringComplete = false;
        }

      }
      else{
        // idle state without any trajectory
        setLegAngles[0] = 2*PI/3;
        setLegAngles[1] = PI/3;
        setLegAngles[2] = 2*PI/3;
        setLegAngles[3] = PI/3;

      }
      
      currentTrajStep = currentTrajStep + 1;
    }
    

    // adjust motor offset
    for (int i=0; i<DXL_ID_CNT; i++){
        setMotorAngles[i] = setLegAngles[i] + PI/4;
    }
    
    // Read potentiometer data for drone height information
    normalizedJoystickHeight = 0;  // (2 * (float)(analogRead(potPin)))/1023.0 - 1;


    // SyncRead
    uint8_t recv_cnt;
    recv_cnt = dxl.syncRead(&sr_infos);
    if(recv_cnt == DXL_ID_CNT){
      for (int i=0; i<DXL_ID_CNT; i++){
        motorAngles[i] = poseUnit*sr_data[i].present_position; // [rad] 
        legAngles[i] = motorAngles[i] - PI/4; //
      }
    }
    // SyncWrite
    // Insert a new Goal Position to the SyncWrite Packet
    for(int i = 0; i < DXL_ID_CNT; i++) {
      sw_data[i].goal_position = (int)(setMotorAngles[i]/poseUnit); // [int RAW position_dynamixel]
    }
    // Update the SyncWrite packet status
    sw_infos.is_info_changed = true;
    
    dxl.syncWrite(&sw_infos);

    if(Timer_02.Tick()){
      if(startTheExperiment){
        if(!isAllScanned ){
          DEBUG_SERIAL.print("<");

          for(int i = 0; i < DXL_ID_CNT; i++) { 
            DEBUG_SERIAL.print(legAngles[i]); 
            if(i < (DXL_ID_CNT)){
              DEBUG_SERIAL.print(",");
            }
            DEBUG_SERIAL.print(setLegAngles[i]); 
            if(i < (DXL_ID_CNT - 1)){
              DEBUG_SERIAL.print(",");
            }
          }
          DEBUG_SERIAL.print(",");
          DEBUG_SERIAL.print(normalizedJoystickHeight); 
          DEBUG_SERIAL.println(">"); 
          
        }
      }   
    }
  }
}
