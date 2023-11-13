# 概率分布模拟动态定价与Q-learning路线推荐，论文题目：Seeking based on Dynamic Prices: Higher Earnings and Better Strategies in Ride-on-demand Services。（TITS）

## 简介

本项目旨在将动态定价作为关键要素，利用概率分布模拟动态定价，并通过强化学习方法Q-learning为网约车司机提供寻客路线推荐。这个系统整合了概率模型和强化学习算法，以优化网约车司机的寻客路线，提高司机长期效益。

## 安装依赖

确保你的环境中已安装以下依赖：

- Python 3.x
- NumPy
- Pandas

## 算法说明

#### 概率分布模拟动态定价：

- 统计的区域平均动态定价，将区域划分为高、中、低三种低价区

- 依据三种定价区的不同定价乘数的概率分布模拟每种区域的动态定价。

#### Q-learning网约车寻客路线推荐

- 利用历史数据，包括订单数据和GPS轨迹数据，建立司机寻客MDP环境

- 利用基于值迭代的强化学习算法q-learning求解MDP最优解。其中包括参数配置，Q表训练等。

## 参数配置

#### 概率分布模拟动态定价

- average_dynamic_price_multiplier: 统计每个网格区域的平均动态定价。
- dynamic_price_pro: 统计高、中、低三种区域的动态价格乘数的频数分布，模拟动态三种区域的动态定价。

#### Q-learning路线推荐

- qlearning_params: Q-learning算法的参数，如学习率、折扣因子等。
- grid_environment_params: 网格环境的参数，包括网格数量、起始点、终点等。

## 注意事项

在使用Q-learning进行路线推荐时，建议通过合理的参数配置和模型训练来获得更好的效果。
动态定价模块和Q-learning模块可以独立使用，也可以结合在一起，根据实际需求进行定制。



## 感谢您使用我们的网约车寻客优化系统！
