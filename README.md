# Self-Driving Car Using OpenCV

### This project explores the implementation of a self-driving car system using Raspberry Pi 4B, coupled with computer vision technologies.

The system employs a Raspberry Pi 4 as the central processing unit, integrating various 
sensors such as cameras to capture the vehicle's surroundings. Computer vision 
algorithms are employed to process the incoming visual data, allowing the self-driving 
car to recognize and interpret its environment. The computer vision pipeline includes 
tasks such as object detection, lane tracking, and obstacle avoidance.
The project involves the integration of motor control mechanisms, allowing the 
Raspberry Pi 4 to send signals to the car's actuators based on the decisions made by the 
computer vision system.

<div>
<img src="https://github.com/mohamedmagdy841/opencv-self-driving-car/assets/64127744/d2331460-c880-4aef-a654-ffde25933e51" width="485" align="Left">
<img src="https://github.com/mohamedmagdy841/opencv-self-driving-car/assets/64127744/8b446a8b-2059-45d9-aa9c-73aaad93cc3b" width="485" align="Right">
</div>
<br clear="right"/>

## Main Idea
* 1. Receiving live images of the road from a camera.
* 2. Images will undergo certain IMAGE PROCESSING operations.
* 3. Values for throttle and steering are determined from images and sensor, allowing car to drive autonomously.

<p align="center">
  <img width="700" src="https://github.com/mohamedmagdy841/opencv-self-driving-car/assets/64127744/37a734be-739b-40d2-8038-de3d8665b8fe">
</p>

## Operations using OpenCV
* 1. Thresholding the Frames.
* 2. Warping the perspective.
* 3. Finding the center of the Road.

## Results

Without obstacle | With obstacle
:-: | :-:
<video src='https://github.com/mohamedmagdy841/opencv-self-driving-car/assets/64127744/efbc3757-8e23-42ff-9888-b5ee97a6fec7' width=180/> | <video src='https://github.com/mohamedmagdy841/opencv-self-driving-car/assets/64127744/706dd157-ffb5-443e-a929-116dce2e9058' width=180/>


