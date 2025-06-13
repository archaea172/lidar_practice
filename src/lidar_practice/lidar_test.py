import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

import numpy as np
from matplotlib import pyplot as plt

class LidarTestNode(Node):
    def __init__(self):
        super().__init__('lidar_test_node')
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/ldlidar_node/scan',
            self.lidar_callback,
            10
        )
        """
        self.ransac_timer_callback = self.create_timer(
            1,
            self.timer_callback
        )"""
        

    def lidar_callback(self, rxdata:LaserScan):
        angle = rxdata.angle_min
        self.point = np.array([
            [0, 0]
        ])
        for range in rxdata.ranges:
            zahyo = np.array([np.cos(angle), np.sin(angle)])*range
            angle += rxdata.angle_increment
            self.point = np.vstack([self.point, zahyo])
            if angle > np.pi:
                break
        

        _, ax = plt.subplots()
        for one in self.point:
            ax.plot(one[0], one[1], '.')
        ax.set_aspect('equal')
        plt.show()

        
    def timer_callback(self):
        pass
        

def main_lidar_test():
    rclpy.init()
    lidar_test_node = LidarTestNode()
    try:
        rclpy.spin_once(lidar_test_node)

    except KeyboardInterrupt:
        pass
    finally:
        lidar_test_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()