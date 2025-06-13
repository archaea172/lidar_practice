import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

import numpy as np
from matplotlib import pyplot as plt

class RansacModel():
    def __init__(self, n:int, delta:float, stop_score:int):
        self.a = None
        self.b = None
        self.c = None
        self.n = n
        self.delta = delta
        self.stop_score = stop_score
        self.max_parameter = None
        self.max_inliers_count = 0

    def cal_parameter(self, point0:np.array, point1:np.array):
        if point0[0] == point1[0]:
            self.a = 1
            self.b = 0
            self.c = point0[0]
            return None
        self.a = (point1[1] - point0[1])/(point1[0] - point0[0])
        self.b = -1
        self.c = point0[1] - self.a*point0[0]

    def cal_inliner(self, datas: np.array):
        d = self.cal_length(datas)
        inliers_index = d < self.delta
        return datas[inliers_index], datas[~inliers_index]
    
    def cal_length(self, points):
        length = np.abs(self.a*points[:, 0] + self.b*points[:, 1] + self.c) / np.sqrt(np.square(self.a) + np.square(self.b))
        return length
    
    def r2(self, inlier_list):
        mu_x = np.mean(inlier_list[0])
        mu_y = np.mean(inlier_list[1])
        var_x = np.var(inlier_list[0])
        cov = np.cov(inlier_list[0], inlier_list[1])

    def predict(self, datas:np.array):
        for i in range(self.n):
            index = np.random.choice(datas.shape[0], size=2, replace=False)
            point0 = datas[index[0]]
            point1 = datas[index[1]]
            self.cal_parameter(point0, point1)
            inlier_list, outlier_list = self.cal_inliner(datas)
            if inlier_list is None:
                continue
            inlier_num = len(inlier_list)
            if inlier_num > self.max_inliers_count:
                self.max_inliers_count = inlier_num
                self.max_parameter = [self.a, self.b, self.c]
                if inlier_num > self.stop_score:
                    print("stop!!:", i)
                    break
        return self.max_parameter, inlier_list, outlier_list

class LidarTestNode(Node):
    def __init__(self):
        super().__init__('lidar_test_node')
        self.ransac_model = RansacModel(100, 1, 100)
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
            zahyo = np.array([np.cos(angle), np.sin(angle)])*range*1000
            angle += rxdata.angle_increment
            self.point = np.vstack([self.point, zahyo])
            if angle > np.pi:
                break
        
        parameter, inlier, outlier = self.ransac_model.predict(self.point)

        edge_point_min = np.amin(inlier, 0).flatten()
        edge_point_max = np.amax(inlier, 0).flatten()
        _, ax = plt.subplots()
        for one in outlier:
            ax.plot(one[0], one[1], '.')
        ax.plot((edge_point_min[0], edge_point_max[0]), (edge_point_min[1], edge_point_max[1]))

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