#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
from std_msgs .msg import Int32

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
VELOCITY_TO_SLOWDOWN_RATIO = 5

# TODO(anyone): these functions should be reused between nodes
# but that's not very easy to do in ROS...
def get_square_dist(d1, d2):
    x2 = d1.x - d2.x
    y2 = d1.y - d2.y
    return x2*x2 + y2*y2

def get_closest_waypoint(pos, waypoints):
    # TODO(anyone): fix it not to return waypoints behind the vehicle
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

def smoothstep(edge0, edge1, x, low_pass = 0.1):
    # Scale, bias and saturate x to 0..1 range
    y = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    if y > low_pass:
        return y
    else:
        return 0

def clamp(x, lowerlimit, upperlimit):
    if x < lowerlimit:
        x = lowerlimit
    if x > upperlimit:
        x = upperlimit
    return x

# get next waypoints starting from idx up to count
def get_next_waypoints(waypoints, idx, count):
    sz = len(waypoints)
    last = idx + count + 1
    if last < sz:
        return waypoints[idx:last]
    else:
        last -= sz # handle circular queue
        return waypoints[idx:] + waypoints[:last]

# get lane from the waypoints with given id
def get_lane_object(id, waypoints):
    lane = Lane()
    lane.header.frame_id = id
    lane.header.stamp = rospy.Time.now()
    lane.waypoints = waypoints
    return lane

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # get max speed
        velocity = rospy.get_param('/waypoint_loader/velocity')
        # times 0.9 to make it slower 
        self.speed_limit = velocity * 0.9 * 1000 / 3600. # m/s 
        rospy.loginfo("param velocity:%s speed_limit:%s m/s", velocity, self.speed_limit)

        # current position of the car
        self.cur_pos = None

        # list of base waypoints
        self.waypoints = None
        self.traffic_waypt_index = None

        rospy.spin()

    def pose_cb(self, msg):
        self.cur_pos = msg.pose.position
        frame_id = msg.header.frame_id
        self.process_waypts(frame_id)

    def process_waypts(self, frame_id):
        if self.waypoints != None and self.cur_pos != None:
            # get the closet waypont to the car
            idx = get_closest_waypoint(self.cur_pos, self.waypoints)
            points = get_next_waypoints(self.waypoints, idx + 1, LOOKAHEAD_WPS)

            # log upcoming 25 waypoints
            # log_vals = [round(wp.twist.twist.linear.x, 1) for wp in points[:25]]
            # rospy.logerr_throttle(10, "v_next: {}".format(log_vals))

            # make lane object
            lane = get_lane_object(frame_id, points)
            # publish  lane
            self.final_waypoints_pub.publish(lane)

    def waypoints_cb(self, lane):
        rospy.logerr("received waypoints: {}".format(len(lane.waypoints)))
        self.waypoints = lane.waypoints

    def traffic_cb(self, msg):
        if self.traffic_waypt_index != msg.data:
            self.traffic_waypt_index = msg.data
            if self.traffic_waypt_index == -1:
                self.fullspeed()
            else:
                self.stop()

    def stop(self):
        # updates waypoints with decceleration to stop before traffic_waypt_index
        idx = get_closest_waypoint(self.cur_pos, self.waypoints)
        if self.traffic_waypt_index < idx:
            # that's a red light behind us, ignore!
            # TODO(anyone): FIX tl_detector logic not to return these!!!
            self.fullspeed()
            return
        rospy.logerr("red light idx: :%s", self.traffic_waypt_index)
        spd = self.speed_limit
        req_slowdown_dist = int(spd * VELOCITY_TO_SLOWDOWN_RATIO)
        deccel_start = max(self.traffic_waypt_index - req_slowdown_dist, 0)
        for i, wp in enumerate(self.waypoints[deccel_start:]):
            wp.twist.twist.linear.x = spd * smoothstep(spd, 0, i * spd / req_slowdown_dist)

    def fullspeed(self):
        if not self.waypoints:
            return
        for wp in self.waypoints:
            wp.twist.twist.linear.x = self.speed_limit

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
