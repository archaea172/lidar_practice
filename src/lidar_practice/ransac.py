import numpy as np
import matplotlib.pyplot as plt


def create_large_dataset_for_ransac(num_inliers=70,
                                    num_outliers=30,
                                    true_a=2.0,
                                    true_b=1.0,
                                    inlier_noise_std_dev=0.5,
                                    x_range=(-10, 10),
                                    outlier_y_multiplier=5, # アウトライアがインライアの直線からどれだけ離れるかの目安
                                    random_seed=42):
    """
    RANSACテスト用のインライアとアウトライアを含む大規模な2Dデータセットを生成します。

    パラメータ:
    - num_inliers (int): 生成するインライアの数。
    - num_outliers (int): 生成するアウトライアの数。
    - true_a (float): インライアが従う直線の真の傾き。
    - true_b (float): インライアが従う直線の真の切片。
    - inlier_noise_std_dev (float): インライアのy座標に加えられる正規分布ノイズの標準偏差。
    - x_range (tuple): インライアとアウトライアのx座標の生成範囲 (min_x, max_x)。
    - outlier_y_multiplier (float): アウトライアを生成する際、真の直線からのずれ具合を調整する係数。
                                     大きいほど直線から離れた位置にアウトライアが生成されやすくなります。
    - random_seed (int): 乱数生成器のシード。Noneにすると実行ごとに異なるデータセットが生成されます。

    戻り値:
    - data (np.ndarray): (num_inliers + num_outliers, 2) の形状を持つデータ点の配列。
    - true_params (dict): 生成に使用した真のパラメータ {'a': true_a, 'b': true_b}。
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    # 1. インライアの生成
    inliers_x = rng.uniform(x_range[0], x_range[1], num_inliers)
    # 真の直線 y = true_a * x + true_b
    inliers_y_exact = true_a * inliers_x + true_b
    # y座標にノイズを追加
    inliers_y = inliers_y_exact + rng.normal(0, inlier_noise_std_dev, num_inliers)
    inliers = np.vstack((inliers_x, inliers_y)).T

    # 2. アウトライアの生成
    outliers_x = rng.uniform(x_range[0], x_range[1], num_outliers)
    # アウトライアは、インライアが従う直線から大きく外れた位置に生成
    # y座標を、真の直線から意図的にずらす
    # (単純な方法として、真の直線からの距離に比例したノイズを加えるか、
    #  あるいは全く異なる範囲で生成する)
    
    # ここでは、真の直線からの距離に大きなランダム値を加えることでアウトライアを生成
    # y_deviation_for_outliers = (rng.random(num_outliers) - 0.5) * 2 * outlier_y_multiplier * (x_range[1] - x_range) * true_a
    # outliers_y = true_a * outliers_x + true_b + y_deviation_for_outliers
    
    # より単純に、y座標を広い範囲でランダムに生成し、インライアの直線から離れやすくする
    y_center_for_outliers = true_a * outliers_x + true_b
    # 符号をランダムにし、一定以上の距離を保つようにオフセットを加える
    random_signs = rng.choice([-1, 1], num_outliers)
    # 直線からの最小距離（目安）をノイズ標準偏差の数倍に設定
    min_dist_from_line = inlier_noise_std_dev * outlier_y_multiplier 
    # y座標のばらつきを大きくする
    y_spread_for_outliers = (x_range[1] - x_range[0]) * abs(true_a) * 0.5 + abs(true_b) + min_dist_from_line
    
    outliers_y = y_center_for_outliers + random_signs * rng.uniform(min_dist_from_line, min_dist_from_line + y_spread_for_outliers , num_outliers)

    outliers = np.vstack((outliers_x, outliers_y)).T

    # 3. インライアとアウトライアを結合
    data = np.vstack((inliers, outliers))

    # 4. データをシャッフル
    rng.shuffle(data)
    
    true_params = {'a': true_a, 'b': true_b}

    return data, true_params

# データセットを生成
# パラメータは自由に変更して試してみてください
num_total_points = 455
inlier_ratio = 0.7 # インライアの割合
n_inliers = int(num_total_points * inlier_ratio)
n_outliers = num_total_points - n_inliers

# データセット生成関数の呼び出し
data_large, true_line_params = create_large_dataset_for_ransac(
    num_inliers=n_inliers,
    num_outliers=n_outliers,
    true_a=4,       # 例：直線の傾き
    true_b=30,       # 例：直線の切片
    inlier_noise_std_dev=0.8, # インライアのばらつき具合
    x_range=(-20, 20),
    outlier_y_multiplier=8, # この値を大きくするとアウトライアがより直線から離れます
    random_seed=123   # シードを固定すると毎回同じデータセットが生成されます
)
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

    
def main():
    ransac_model = RansacModel(100, 1, 100)
    parameter, inlier, outlier = ransac_model.predict(data_large)
    
    _, ax = plt.subplots()
    # _, ax2 = plt.subplots()

    edge_point_min = np.amin(inlier, 0).flatten()
    edge_point_max = np.amax(inlier, 0).flatten()
    ax.plot(edge_point_min[0], edge_point_min[1], marker='.')
    ax.plot(edge_point_max[0], edge_point_max[1], marker='.')
    ax.plot((edge_point_max[0], edge_point_min[0]), (edge_point_max[1], edge_point_min[1]))
    # ax.plot(edge_point_min)
    for one in outlier:
        ax.plot(one[0], one[1], marker='.')
        pass
    
    for one in data_large:
        # ax.plot(one[0], one[1], marker='.')
        pass

    

    print("インライア")
    print(inlier.shape)
    print("アウトライア")
    print(outlier.shape)
    print("推定")
    print(parameter[0], parameter[1], parameter[2])
    print("正解")
    print(true_line_params)
    plt.show()

if __name__ == '__main__':
    main()