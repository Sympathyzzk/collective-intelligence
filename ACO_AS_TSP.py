#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author: Sympathyzzk

import numpy as np
import copy
import sys
import tkinter  # //GUI模块
from functools import reduce

# 参数

'''
ALPHA:信息素的重要程度。值越大，则蚂蚁选择之前走过的路径可能性就越大，
      值越小，则蚁群搜索范围就会增大，不容易陷入局部最优。但是收敛速度会降低

BETA:启发信息的重要程度。Beta值越大，蚁群越就容易选择局部较短路径（启发信息是距离的反比），
     这时算法收敛速度会加快，但是随机性不高，容易得到局部的相对最优。

RHO：信息素挥发速率
'''

(ALPHA, BETA, RHO, Q) = (1.0, 2.0, 0.5, 100.0)

# 城市数，蚁群
(city_num, ant_num) = (50, 50)

distance_x = [
    178, 272, 176, 171, 650, 499, 267, 703, 408, 437, 491, 74, 532,
    416, 626, 42, 271, 359, 163, 508, 229, 576, 147, 560, 35, 714,
    757, 517, 64, 314, 675, 690, 391, 628, 87, 240, 705, 699, 258,
    428, 614, 36, 360, 482, 666, 597, 209, 201, 492, 294]

distance_y = [
    170, 395, 198, 151, 242, 556, 57, 401, 305, 421, 267, 105, 525,
    381, 244, 330, 395, 169, 141, 380, 153, 442, 528, 329, 232, 48,
    498, 265, 343, 120, 165, 50, 433, 63, 491, 275, 348, 222, 288,
    490, 213, 524, 244, 114, 104, 552, 70, 425, 227, 331]

# 城市距离为0 信息素为1  50x50
distance_graph = [[0.0 for col in range(city_num)] for raw in range(city_num)]
pheromone_graph = [[1.0 for col in range(city_num)] for raw in range(city_num)]


# ----------- 蚂蚁 -----------
class Ant(object):

    # 初始化单个蚂蚁的出生点和编号
    def __init__(self, ID):
        self.ID = ID  # ID：蚂蚁的编号
        self.__clean_data()  # 随机初始化出生点,初始化其他信息

    # 初始数据
    def __clean_data(self):

        # 初始化所有信息
        self.path = []  # 当前蚂蚁的路径
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)]  # 探索城市的状态，禁忌表

        # 随机初始出生点 0<= <=city_num - 1并设置路径以及禁忌表状态
        city_index = np.random.randint(0, city_num - 1)
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city(self):

        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率

        # 非禁忌表中的所有城市的概率和
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in range(city_num):
            if self.open_table_city[i]:

                try:

                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        (1.0 / distance_graph[self.current_city][i]), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID=self.ID,
                                                                                                current=self.current_city,
                                                                                                target=i))
                    sys.exit(1)

        # 使用的是AS方法，只根据概率进行选择
        prob = np.array(select_citys_prob) / total_prob
        next_city = np.random.choice(range(city_num), size=1, p=prob)[0]  # prob对应allow_list中的概率

        # 未从概率产生，顺序选择一个未访问城市
        if (next_city == -1):
            next_city = np.random.randint(0, city_num - 1)

            while (self.open_table_city[next_city]) == False:  # if==False,说明已经遍历过了
                next_city = np.random.randint(0, city_num - 1)

        # 返回下一个城市序号
        return next_city

    # 计算路径总距离
    def __cal_total_distance(self):
        temp_distance = 0.0
        for i in range(1, city_num):
            start, end = self.path[i], self.path[i - 1]
            temp_distance += distance_graph[start][end]

        # 构成回路
        end = self.path[0]
        temp_distance += distance_graph[start][end]

        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_city):
        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]

        # 准备下一次移动
        self.current_city = next_city
        self.move_count += 1

    # 搜索路径
    def search_path(self):
        # 初始化数据
        self.__clean_data()

        # 搜素路径，遍历完所有城市为止
        while self.move_count < city_num:
            # 移动到下一个城市
            next_city = self.__choice_next_city()
            self.__move(next_city)

        # 计算路径总长度
        self.__cal_total_distance()


# ----------- TSP问题 -----------


class TSP(object):

    def __init__(self, root, width=800, height=600, n=city_num):
        # 创建画布
        self.root = root
        self.width = width
        self.height = height

        # 城市数目初始化为city_num
        self.n = n

        # tkinter.Canvas
        self.canvas = tkinter.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#EBEBEB",  # 背景白色
        )

        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
        self.title("TSP蚁群算法(n:初始化 e:开始搜索 s:停止搜索 q:退出程序)")
        self.__r = 5
        self.__bindEvents()  # 绑定按键以及对应的事件
        self.new()  # 标出点的图像和对应坐标 初始化种群和信息素（1）

        # 计算城市之间的距离
        for i in range(city_num):
            for j in range(city_num):
                temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[i][j] = float(int(temp_distance + 0.5))

    # 按键响应程序
    def __bindEvents(self):
        self.root.bind("q", self.quite)  # 退出程序
        self.root.bind("n", self.new)  # 初始化
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)  # 停止搜索

    # 更改标题
    def title(self, s):
        self.root.title(s)

    # 初始化
    def new(self, evt=None):
        # 停止线程
        self.__running = False

        self.clear()  # 清除信息

        self.nodes = []  # 节点坐标
        self.nodes2 = []  # 节点对象

        # 初始化城市节点
        for i in range(len(distance_x)):
            # 在画布上随机初始坐标
            x = distance_x[i]
            y = distance_y[i]
            self.nodes.append((x, y))

            # 生成节点圆对象，半径为self.__r
            node = self.canvas.create_oval(x - self.__r, y - self.__r, x + self.__r, y + self.__r,
                                           fill="#00ffff",  # 填充色彩
                                           outline="#000000",  # 轮廓白色
                                           tags="node",
                                           )
            self.nodes2.append(node)

            # 显示坐标
            self.canvas.create_text(x, y - 10,  # 使用create_text方法在坐标（x, y - 10）处绘制文字
                                    text='(' + str(x) + ',' + str(y) + ')',  # 所绘制文字的内容
                                    fill='black'  # 所绘制文字的颜色为黑色
                                    )

        # 初始城市之间的距离和信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = 1.0

        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1)  # 初始最优解
        self.best_ant.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数

    # 将节点按order顺序连线
    def line(self, order):
        # 删除原线
        self.canvas.delete("line")

        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill="#000000", tags="line")
            return i2

        # order[-1]为初始值
        reduce(line2, order, order[-1])

    # 清除画布
    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

    # 退出程序
    def quite(self, evt=None):
        self.__running = False

        self.root.destroy()
        print(u"\n程序已退出...")
        sys.exit()

    # 停止搜索
    def stop(self, evt=None):
        self.__running = False

    # 开始搜索
    def search_path(self, evt=None):
        # 开启线程
        self.__running = True

        while self.__running:
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path()

                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)

            # 更新信息素
            self.__update_pheromone_gragh()

            print(u"迭代次数：", self.iter, u"最佳路径总距离：", int(self.best_ant.total_distance))

            # 连线
            self.line(self.best_ant.path)

            # 设置标题
            self.title("TSP蚁群算法(n:随机初始 e:开始搜索 s:停止搜索 q:退出程序) 迭代次数: %d" % self.iter)

            # 更新画布
            self.canvas.update()

            self.iter += 1

    # 更新信息素
    def __update_pheromone_gragh(self):
        # 获取每只蚂蚁在一次迭代中在路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        for ant in self.ants:
            for i in range(1, city_num):
                start, end = ant.path[i - 1], ant.path[i]

                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]

    # 主循环
    def mainloop(self):
        self.root.mainloop()


# ----------- 程序的入口处 -----------


if __name__ == '__main__':
    TSP(tkinter.Tk()).mainloop()
