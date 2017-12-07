# Team Drive Directive

## Members
Name                       | eMail                             | Time Zone
---------------------------|-----------------------------------|----------
Wade Ju                    | wade_ju@me.com (wadeju@yahoo.com) | UTC-8
Jan Grodowski	           | jgrodowski@gmail.com              | UTC+1
Christian Reiser	   | reisercg@gmail.com	               | UTC+1
Cedric Bodet               | bodetc@gmail.com	               | UTC+1/+2
Piotr (Peter) Szczesnowicz | szczesnowicz.piotr@gmail.com      | UTC-05

## Summary
This is our submission for the Udacity Self-Driving Car Nanodegree System Integration (Capstone) Project. The goal of this project was to implement ROS nodes to control a simulated and real car around a track given the waypoints and to detect and classify traffic lights given a video feed.

## Conclusion
We have successfully implemented the following ROS nodes: Waypoint Updater, Drive-By-Wire (DBW), and Traffic Light Detection.

To improve our system we would also need to detect obstacles like pedestrians, bicycles, motorcycles, and cars.

## Waypoint Updater Node
This node  subscribes to `/base_waypoints`, `/current_pose`, and `/traffic_waypoint`, and publishes 200 waypoints with velocities from the current car position to `/final_waypoints`.

* `/base_waypoints`: list of waypoints with positions in world coordinates
* `/current_pose`: current car position in world coordinates
* `/traffic_waypoint`: the waypoint index of traffic light stop lines 

We get maximum velocity from `/waypoint_loader/velocity` and convert it to meters per second.

When a new `/current_pose` topic comes in, `pose_cb` callback is invoked to get an updated position of the car and calculate new waypoints from the updated position. We get next 200 waypoints from the base waypoints and wrap around if the end is reached.

When a new traffic light topic comes in, `traffic_cb` callback is invoked to adjust speed. First, we check whether it's a new message to avoid duplication. For red light, we use a distance equal to max speed times five to slow down velocity of the car linearly before a stop line.  

## DBW Node
Once messages are being published by the waypoint updater node to `/final_waypoints`, the vehicle's waypoint follower will publish twist commands to the `/twist_cmd` topic, which contains linear and angular velocities.

The DBW node subscribed to the `/twist_cmd` topic and uses various controllers to provide appropriate throttle, brake, and steering commands:

* `/vehicle/throttle_cmd`
* `/vehicle/brake_cmd`
* `/vehicle/steering_cmd`

DBW uses a PID control to calculate throttle, brake, steering angle with input of target linear and angular velocity, and the current velocity. Throttle needs to be limited between 0.0 and 1.0. Acceleration needs to be limited to parameters of deceleration and acceleration limitations. 

Brake is calculated by the formula of (vehicle mass + fuel capacity * density) * acceleration * wheel radius.
If brake is less the brake dead band parameter, we simply set it to zero.
Steering angle is calculated by the yaw controller provided by Udacity.

If a safety driver takes over control of the car, the DBW node will be disabled.

## Traffic Light Detection Node
The traffic light detection node detects traffic light color from `/image_color`, an image stream from the car's camera.
The node publishes the index of the waypoint for nearest upcoming red light's stop line to a single topic `/traffic_waypoint`. For example, if waypoints is the complete list of waypoints, and an upcoming red light's stop line is nearest to waypoints[12], then 12 should be published through `/traffic_waypoint`. This index can later be used by the waypoint updater node to set the target velocity for waypoints[12] to 0 and smoothly decrease the vehicle velocity in the waypoints leading up to waypoints[12].

We have gone through three implementations for this module. First, we implemented it with `/vehicle/traffic_lights` topic from simulator. It has both color and position of traffic lights. It helped us to develop and test solution.  

Second, we implemented a HSV classifier to detect image of red light. We get positions of stop lines from the provided config files. When an image comes in, `image_cb` callback function is invoked to process an image. If the state sustains more three threshold count, we publish waypoint index of the corresponding stop line for red light to `/traffic_waypoint`. Otherwise, we publish -1 to `/traffic_waypoint`.

Third, we used trained a deep learning network and used it to detect and classify traffic light colors.

## Traffic Light Detection Classifier

### HSV Classifier
HSV (Hue, Saturation, Value) is better than RGB for this kind of classification.

We define a low and high range of red in HSV:

```
RED_MIN = np.array([0, 180, 180], np.uint8)
RED_MAX = np.array([10, 255, 255], np.uint8)
THRESHOLD = 60
```

First, we use OpenCV to convert image from BGR space to HSV space:

```
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

Second, we use OpenCV to change how many pixels are in the red range:

```
frame_threshed = cv2.inRange(img_hsv, self.RED_MIN, self.RED_MAX)
```

If it's higher than the threshold, it's a red light:

```
count = cv2.countNonZero(frame_threshed)
if  (count > self.THRESHOLD):
	return  TrafficLight.RED 
else:
	return  TrafficLight.UNKNOWN
```

### Deep Learning Network Classifier

#### Network
The deep learning network classifier is the Single Shot Detection (SSD) Mobilenet V1 network. This network is part of the [Tensorflow Model Zoo](https://github.com/tensorflow/models) which is a collection of pre-trained deep learning networks. We chose the SSD Mobilenet V1 network because of its realtime inference performance on mobile devices, such as a self-driving car.

#### Dataset
The training dataset consisted of:
* 180 images of red traffic lights extracted from a rosbag file recording of the Udacity test track
* 60 images of red traffic lights (each with approximately 3 traffic lights) captured from the Udacity simulator
* Image sizes of 800x640 pixels

<img src="https://raw.githubusercontent.com/pszczesnowicz/SDC-P13-Capstone-Drive-Directive/master/readme_images/train_track_red0.jpg" width="400"><img src="https://raw.githubusercontent.com/pszczesnowicz/SDC-P13-Capstone-Drive-Directive/master/readme_images/train_sim_red0.jpg" width="400">

#### Results
After training on the aforementioned dataset for 400 steps, the network made the following correct detections:

<img src="https://raw.githubusercontent.com/pszczesnowicz/SDC-P13-Capstone-Drive-Directive/master/readme_images/test_track_red0.jpg" width="400"><img src="https://raw.githubusercontent.com/pszczesnowicz/SDC-P13-Capstone-Drive-Directive/master/readme_images/test_sim_red0.jpg" width="400">

## Simulator Images
<img src="https://raw.githubusercontent.com/pszczesnowicz/SDC-P13-Capstone-Drive-Directive/master/readme_images/sim_green0.jpg" width="800">

<img src="https://raw.githubusercontent.com/pszczesnowicz/SDC-P13-Capstone-Drive-Directive/master/readme_images/sim_red0.jpg" width="800">

<img src="https://raw.githubusercontent.com/pszczesnowicz/SDC-P13-Capstone-Drive-Directive/master/readme_images/sim_none0.jpg" width="800">

## References

[Udacity Self-Driving Car ND](http://www.udacity.com/drive)

[Udacity Self-Driving Car ND - Capstone Project Repo](https://github.com/udacity/CarND-Capstone)

[Supercharge your Computer Vision models with the TensorFlow Object Detection API](https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html)
