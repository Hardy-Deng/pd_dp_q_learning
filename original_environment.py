# 导入库信息
import time
import numpy as np
import pandas as pd

# 设定环境信息

Max_episode = 1000000  # 设置学习函数的学习轮次
SPEND_TIME = 60  # 设置每次探索时间

data = pd.read_csv("./环境/拾取概率.csv", encoding='UTF-8')       # 在状态s的拾取概率
grid = pd.read_csv("./环境/调理数据4.csv", encoding='UTF-8')      # 起始地到目的地的平均时间和平均距离
matri = np.array(pd.read_csv("./环境/TJ.csv", encoding='UTF-8'))  # 状态转移概率矩阵
dy = pd.read_csv("./环境/dy.csv", encoding='UTF-8')               # 状态s的平均动态系数


# 创建一个网格环境
class Orignal_Env:
    def __init__(self, start_l, reward_dy=1):
        # 定义刚开始时候的状态
        self.start_state = start_l
        # 定义刚开始时间装态
        self.start_time = 0
        # 动作空间映射为列表
        self.actions = [ac for ac in range(9)]
        # 是否启用动态定价【reward_dy为1启用】
        self.reward_dy = reward_dy
        # 定义当前状态(暂时定为开始状态）
        self.currentstate = self.start_state
        self.currenttime = 0

    # 环境状态改变，方便频繁变换环境跑单次收益
    def environment_change(self, change_start_id):
        self.start_state = change_start_id
        self.currentstate = self.start_state

    def reset(self):
        # 重置当前状态为开始状态
        self.currentstate = self.start_state
        # 返回初始状态
        return self.start_state

    def get_next_grid(self, action):
        now_grid = self.currentstate
        x, y = self.tran_id_gridpositon(now_grid)
        if action == 1:
            x -= 1
            y -= 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x -= 1
            y += 1
        elif action == 4:
            y += 1
        elif action == 5:
            return now_grid
        elif action == 6:
            y -= 1
        elif action == 7:
            x += 1
            y -= 1
        elif action == 8:
            x += 1
        elif action == 9:
            x += 1
            y += 1
        if 0 < x < 31 and 0 < y < 31:
            return 30 * (x - 1) + y
        else:
            return 0

    #  将网格划分为低中高价区：
    def grid_divide(self):
        high_list = []
        mid_list = []
        low_list = []
        for i in range(len(dy)):
            s_id = dy.iloc[i, 0]
            conf = dy.iloc[i, 5]
            if conf <= 1.2:
                low_list.append(s_id)
            if conf >= 1.5:
                high_list.append(s_id)
            if 1.5 > conf > 1.2:
                mid_list.append(s_id)
        return low_list, mid_list, high_list

    def find_index(self, l, t, action):
        index_l = l - 1
        index_t = t - 1
        index_action = action - 1
        return index_l, index_t, index_action

    # 在网格l找到乘客的概率
    def find_prob(self, l):
        return data.iloc[l - 1, 3]

    #  模拟动态定价波动的概率方法。。已知其实id号和距离模拟动态定价机制返回动态乘数和奖励
    def prob_model(self, new_id, new_distance):
        # 动态定价
        if self.reward_dy:
            low_list, mid_list, high_list = self.grid_divide()
            dfconf = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
            prob_highconf = [0.1, 0.1, 0.1, 0.1, 0.1, 0.10, 0.4]
            prob_midconf = [0.25, 0.05, 0.1, 0.03, 0.17, 0.07, 0.33]
            prob_lowconf = [0.5, 0.05, 0.1, 0.05, 0.1, 0.1, 0.1]
            if new_id in high_list:
                index = np.random.choice(dfconf, p=prob_highconf)
            if new_id in mid_list:
                index = np.random.choice(dfconf, p=prob_midconf)
            if new_id in low_list:
                index = np.random.choice(dfconf, p=prob_lowconf)
            reward = index * (15 + 2.8 * new_distance)
        # 非动态定价
        else:
            reward = 1.0 * (15 + 2.8 * new_distance)
        return reward

    # 未寻找到乘客的奖励
    def reward_0(self, l, t, action, seek_time):
        if action == 1 or action == 3 or action == 7 or action == 9:
            reward = -0.5 * 1.4
            t = t + 2
            seek_time += 2
        else:
            reward = -0.5 * 1
            t = t + 1
            seek_time += 1
        return l, t, reward, seek_time  # 没有接到乘客，司机还在网格l

    # 判断找到乘客的情况，在l找到乘客的情况，返回下一个地址，时间，奖励；
    def reward_1(self, l, t, action, seek_time, trans_time):
        # 到下一个id的时空花销
        if action == 1 or action == 3 or action == 7 or action == 9:
            cost_l = -0.5 * 1.4
            t = t + 2
            seek_time += 2
        else:
            cost_l = -0.5 * 1
            t = t + 1
            seek_time += 1
        prob = []
        ID = []
        for i in range(0, 900):  # 下标
            ID.append(i)
        ID = np.array(ID)
        for j in range(0, 900):
            prob.append(matri[l - 1][j])
        prob = [round(i, 3) for i in prob]  # 取三位小数
        prob = np.array(prob)
        index = np.random.choice(ID, p=prob.ravel())  ###    数组的下标，选择下一跳转的地址
        ##  新的ID， 是已经数据索引上+1了 所以就是当前网格的ID
        new_id = index + 1
        demo = grid[grid['s_id'] == l]
        cost_t = 0
        for i in range(len(demo)):
            e_id = demo.iloc[i, 1]
            if e_id == new_id:
                cost_t = demo.iloc[i, 3]
                new_distance = demo.iloc[i, 4]
        reward = self.prob_model(l, new_distance)
        new_t = t + cost_t
        trans_time += cost_t
        reward += cost_l
        return new_id, new_t, reward, seek_time, trans_time

    # 将将网格id转化为网格中的坐标，左边原点为左下角
    def tran_id_gridpositon(self, gridid):
        l = gridid - 1
        x = int(l / 30) + 1
        y = l - (x - 1) * 30 + 1
        return [x, y]

    # 每步探索函数，加入寻客时间和运送时间的统计
    def step_amend(self, action, t, seek_time, trans_time):
        # 移动后的状态定义为s_
        s_ = self.get_next_grid(action)
        # 更新环境状态
        self.currentstate = s_
        # 在下一个id可以分为两种情况：接到乘客，没有接到乘客
        prob = self.find_prob(s_)
        flag_find = np.random.rand()
        if flag_find < prob:  # 在下一个id接到乘客
            new_l, new_t, reward, seek_time, trans_time = self.reward_1(s_, t, action, seek_time, trans_time)
            self.currentstate = new_l
            self.currenttime = new_t
        else:  # 接不到乘客
            new_l, new_t, reward, seek_time = self.reward_0(s_, t, action, seek_time)
            self.currentstate = new_l
            self.currenttime = new_t
        if new_t >= SPEND_TIME:
            done = 1
        else:
            done = 0
        return new_l, new_t, reward, done, seek_time, trans_time

