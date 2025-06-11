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
        self.range = rxdata.ranges
        
        for i in range(len(self.range)):
            try:
                angle = 2*np.pi/len(self.range)*i
                zahyo = np.array([np.cos(angle), np.sin(angle)])*self.range[i]*100
                self.point[i] = zahyo.copy

            except:
                continue
        
    def timer_callback(self):
        img = np.full((700, 700, 3), 128, dtype=np.uint8)
        
        for i in range(455):
            # cv2.circle(img, self.point[i], 1, (255, 0, 0), -1)
        # cv2.imshow("lidar", img)
            pass
        self.get_logger().info(str(self.point[0]))
        cv2.waitKey(3)
        

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