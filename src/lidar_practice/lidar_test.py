import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

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
        

    def lidar_callback(self, rxdata:LaserScan):
        self.range = rxdata.ranges
        self.get_logger().info(str(self.range[226]))
        

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