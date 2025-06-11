import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

import numpy as np
import cv2

class LidarTestNode(Node):
    def __init__(self):
        super().__init__('lidar_test_node')
        self.range = []

        self.lidar_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10
        )

        self.ransac_timer_callback = self.create_timer(
            0.1,
            self.timer_callback
        )
        self.point = np.ndarray((455, 2))
        

    def lidar_callback(self, rxdata:LaserScan):
        angle = rxdata.angle_min
        self.point_zahyo = np.ndarray((len(rxdata.ranges), 2))
        for point in rxdata.ranges:
            zahyo = np.array([np.cos(angle), np.sin(angle)])*point
            angle += rxdata.angle_increment

        self.get_logger().info(str(zahyo[224]))
        
    def timer_callback(self):
        pass
        

def main_lidar_test():
    rclpy.init()
    lidar_test_node = LidarTestNode()
    try:
        rclpy.spin(lidar_test_node)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_test_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()