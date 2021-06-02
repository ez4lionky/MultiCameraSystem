# MultiCameraSystem
#### Synchronize multiple (greater than 4) cameras with Realsense T265 tracking.

You can plug multiple cameras by one hub into your PC, before run the code, 
you need to ensure the device id is correct. While running the code, 
program will record realtime streams from all devices, including the tracking
data (camera pose) from Realsense T265. In the final, it will synchronize
color streams from webcam and tracking data offline.

**Note:** The FPS of webcam is limited to under 30 FPS.