import numpy as np
import pandas as pd
import time as ostime
import original_environment as oe

# np.random.seed(1)  # 设置随机数种子，方便重复实验观察效果
Max_episode = 10000
LEARNING_RATE = 0.01
GAMA = 0.7
EPSILON = 0.3  # 贪心概率
SPEND_TIME = 60  # 设置每次探索时间


# Q表
class Q:
    def __init__(self,
                 epsilon=EPSILON,
                 learning_rate=0.01,  # 学习率
                 gamma=0.7,  # 折扣因子
                 q_table=None):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = q_table
        if q_table is None:
            self.create_q_table()

    # 初始化Q表， Q表都存储在一个数组内。Q表为所有时间的q表集合
    def create_q_table(self):
        index = pd.MultiIndex.from_product([[i for i in range(1, 901)],
                                            [i for i in range(0, 60)]])
        q_table = pd.DataFrame(columns=range(1, 10), data=np.zeros([54000, 9]),
                               index=index)
        q_table.index.names = ["grid_id", "minute"]
        self.q_table = q_table

    # 随机选择qt中最大值中的一个返回
    def random_select_max(self, qt):
        max_value = max(qt)
        max_index = []
        for i in range(9):
            if qt[i] == max_value:
                max_index.append(i + 1)
        return np.random.choice(max_index)

    # 在时间t状态l是动作选择，返回动作
    def choose_action(self, state, t):
        epsilon = self.epsilon
        actions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if np.random.rand() < epsilon:
            # 贪婪策略随机探索动作
            action = np.random.choice(actions)
        else:
            qt = np.array(self.q_table.loc[(state, t)].values)
            action = self.random_select_max(qt)
        return action

    # q-learning
    def learn(self, state, t, action, reward, next_state, t_, done):
        learning_rate = self.learning_rate  # 学习率
        gama = self.gamma  # 折扣因子
        #  二维数据的贝尔曼方程的运算。
        # 估计q值
        q_predict = self.q_table.at[(state, t), action]
        # 实际q值
        if done:
            next_state_qvalue = np.zeros(9)
        else:
            next_state_qvalue = self.q_table.loc[(next_state, t_)].values
        q_target = reward + (1 - done) * gama * max(next_state_qvalue)
        self.q_table.at[(state, t), action] += learning_rate * (q_target - q_predict)


# q_leaning训练q表
def update_q_leaning(q_table, env, episode=Max_episode):
    """
    :param q_table: Q类
    :param env: 环境类
    :return:
    """
    for i_episode in range(episode):
        # 随机初始化位置训练q表
        s_id = np.random.randint(1, 901)
        env.environment_change(s_id)
        t = 0
        trans_time = 0
        seek_time = 0
        while True:
            # 先依靠Q_learning探索和利用产生一个动作
            action = q_table.choose_action(s_id, t)
            # 当执行动作action后跳出网格区域，重新选择动作
            next_id = env.get_next_grid(action)
            while next_id > 900 or next_id < 1:
                action = q_table.choose_action(s_id, t)
                next_id = env.get_next_grid(action)
            s, t_, r, done, seek_time, trans_time = env.step_amend(action, t, seek_time, trans_time)
            t_ = int(t_)
            # 更新Q表
            q_table.learn(s_id, t, action, r, s, t_, done)
            # 更新状态
            s_id = s
            t = t_
            if done:
                break
        if i_episode % 100 == 0:
            print("Q表训练到第{}次轮次".format(i_episode))


# simulation
def driver_simulation(q_table, env, starting_list):
    """
    :param q_table: Q类
    :param env: 环境类
    :param starting_list: 司机的起始点
    :return:
    """
    # 本轮次的总收益
    score = []
    # 本轮次的接单数
    n_orders = []
    # 本轮次的每单平均收益
    peer_order_revenue = []
    # 本轮次单位是时间的收益（每60分钟）
    peer_hour_revenue = []
    # 每一轮次的有效载客时间和寻客时间
    sum_trans_time = []
    sum_seek_time = []
    for i_episode in range(len(starting_list)):
        # 初始化司机起始位置
        s_id = starting_list[i_episode]
        env.environment_change(s_id)
        r_sample = []
        t = 0
        trans_time = 0
        seek_time = 0
        while True:
            # 先依靠Q_learning探索和利用产生一个动作
            action = q_table.choose_action(s_id, t)
            # 当执行动作action后跳出网格区域，重新选择动作
            next_id = env.get_next_grid(action)
            while next_id > 900 or next_id < 1:
                action = q_table.choose_action(s_id, t)
                next_id = env.get_next_grid(action)
            s, t_, r, done, seek_time, trans_time = env.step_amend(action, t, seek_time, trans_time)
            r_sample.append(r)
            # 更新Q表
            q_table.learn(s_id, t, action, r, s, t_, done)
            # 更新状态
            s_id = s
            t = t_
            # 计算本轮次的所有数据
            if done:
                print("q_learning_strategy Episode {} finished after {} mins".format(i_episode, t + 1))
                score.append(sum(r_sample))
                print("sum revenue", sum(r_sample))
                n_orders.append(sum(i > 0 for i in r_sample))
                if n_orders[i_episode]:
                    peer_order_revenue.append(score[i_episode] / n_orders[i_episode])
                else:
                    peer_order_revenue.append(0)
                peer_hour_revenue.append(score[i_episode] / (t / 60))
                sum_seek_time.append(seek_time)
                sum_trans_time.append(trans_time)
                break
    return [score, n_orders, peer_order_revenue, peer_hour_revenue, sum_seek_time, sum_trans_time]


# 将强化学习算法所返回的所有轮次记录进行处理
def recording(episode_data):
    dataFrame = pd.DataFrame(np.array(episode_data).T,
                             columns=["score", 'n_orders', 'peer_order_revenue', 'peer_hour_revenue',
                                      'sum_seek_time', 'sum_trans_time'])
    dataFrame["average_profit"] = dataFrame["score"] / dataFrame["sum_trans_time"]
    dataFrame.loc[dataFrame.index[dataFrame["sum_trans_time"] == 0], "average_profit"] = 0
    dataFrame["revenue_efficiency"] = dataFrame["score"] / (dataFrame["sum_seek_time"] + dataFrame["sum_trans_time"])
    dataFrame["utilization_rate"] = dataFrame["sum_trans_time"] / (
            dataFrame["sum_seek_time"] + dataFrame["sum_trans_time"])
    dataFrame["average_profit"] = dataFrame["score"] / dataFrame["sum_trans_time"]
    dataFrame.loc[dataFrame.index[dataFrame["sum_trans_time"] == 0], "average_profit"] = 0
    dataFrame["revenue_efficiency"] = dataFrame["score"] / (dataFrame["sum_seek_time"] + dataFrame["sum_trans_time"])
    dataFrame["utilization_rate"] = dataFrame["sum_trans_time"] / (dataFrame["sum_seek_time"] + dataFrame["sum_trans_time"])
    return dataFrame


if __name__ == '__main__':
    # 对程序运行时间计时
    start_time = ostime.perf_counter()

    # 训练q表，随机起点运行10000次：
    q = Q()
    env = oe.Orignal_Env(250, 1)
    update_q_leaning(q, env, 1000000)

    # 司机仿结果统计
    s = [163, 186, 193, 200, 228, 230, 233, 239, 245, 253, 258, 279, 280, 286, 288, 291, 305, 312, 314, 317, 318,
         319, 321, 322, 323, 341, 342, 343, 347, 348, 349, 351, 352, 353, 368, 370, 371, 372, 374, 377, 379,
         380, 381, 385, 400, 407, 408, 409, 410, 429, 431, 432, 434, 438, 440, 441, 458, 459, 469, 471, 472,
         486, 491, 498, 500, 500, 501, 502, 517, 524, 524, 527, 527, 529, 550, 552, 553, 561, 565, 579, 579,
         581, 588, 589, 589, 593, 611, 617, 623, 623, 627, 653, 682, 750]
    recording(driver_simulation(q, env, s)).to_csv(r"C:\data\代码\its_cqu\recording.csv")

    end_time = ostime.perf_counter()
    print("运行时间为：", round(end_time - start_time), "seconds")
    pass
