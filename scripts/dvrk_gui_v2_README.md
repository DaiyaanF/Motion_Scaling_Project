Files to compile: 
dvrk_gui_v2.py
attribute_manager.py
analyze_comprehensive.py
dynamic_patterns.py

Trajectory patterns: 
1. Dot
2. Circle
3. Line 
4. Figure8
5. Triganle

Dot is for practicing only, It may be included in the data analysis for basline assessment (what is the minimal perceivable accracy for the individual user)

Workflow:

1. Start dvrk_gui_v2.py

2. Initialize MTM, wait for 5 sec, AMBF, teleoperation

3. Config the experiemnet
    To configure the experiement, first provide am ID for the configuration set. Then hit the button below it.
    You will be able to set trajectory types, size, motion scaling factors, rotations variations, position variations, and repetition number.
    The configuration will be stored in a json file, see output terminal for information

4. To start the data collection pipeline, hit start
    There will be N+1 trials, where N is the combinations you specified before, 1 is the dot pattern of size 10, positioned in the center, motion scaling factor =1. 
    The dot pattern will always appear first, while the other combinations are randomized for the experiemnet. It serves as the parctice session for the user to adjust for comfort. 
    The real-time psm2 dist plot will start running once you hit start. You can use the dot as the indicator and tell the user if they are getting closer or further.
    Once the camera foot pedal is pressed down, the actual experiement will start. Press the foot pedal again to switch combinations. After the last combination is done, press the foot pedal to shutdown the pipeline and close the ros bag.
    You will be able to stop the data collection pipeline at anytime if you hit the red stop button. Hit start again to start another session under the same configuration. 

5. The ros bag will be stored under the comprehensie_data/ID/timestamp, where timestamp is used to distinguish individual seesions with the same configuration.
    The bag will contain video, timestamp, ID, psm1_dist, psm2_dist, clutch count, footpedal count, combination number, repetition number, shape, size, motion scale, rotation as roll pitch yaw, position as xyz, current index, and total combination number(N+1)