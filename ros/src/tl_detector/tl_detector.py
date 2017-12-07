#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier_ml import TLClassifier

import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 2


# TODO: these functions should be reused between nodes
# but that's not very easy to do in ROS...
def get_square_dist(d1, d2):
    x2 = d1.x - d2.x
    y2 = d1.y - d2.y
    return x2 * x2 + y2 * y2


def get_closest_waypoint(pos, waypoints):
    if waypoints != None:
        idx = 0
        dist = get_square_dist(pos, waypoints[0].pose.pose.position)
        for i in range(1, len(waypoints)):
            waypt = waypoints[i]
            d = get_square_dist(pos, waypt.pose.pose.position)
            if (d < dist):
                idx = i
                dist = d
        return idx
    else:
        return -1


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_lines = map(
            lambda line_pos: Pose(Point(line_pos[0], line_pos[1], 0), None),
            self.config['stop_line_positions']
        )

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, lane):
        self.waypoints = lane.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            # rospy.loginfo("publish light_wp:%s", light_wp)
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
        if self.light_classifier == None:
            return TrafficLight.RED
        else:
            return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # TODO: classification should not be performed as part of image_cb, because it
        # has delays. Wrap this into a Rate object later on.
        # probably we can limit to process image only within certain distance of a light

        light_wp = -1
        idx = -1

        # List of positions that correspond to the line to stop in front of for a given intersection
        if (self.pose):
            car_position = get_closest_waypoint(self.pose.pose.position, self.waypoints)
            stop_line_positions_dists = map(
                lambda pos: get_square_dist(self.pose.pose.position, pos.position),
                self.stop_lines
            )
            idx = stop_line_positions_dists.index(min(stop_line_positions_dists))

        if idx >= 0:
        	state = self.get_light_state()
        	light_wp = get_closest_waypoint(self.stop_lines[idx].position, self.waypoints)
        	return light_wp, state
        else:
        	return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
