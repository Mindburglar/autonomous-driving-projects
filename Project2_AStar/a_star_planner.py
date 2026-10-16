"""
Project 2: A* Path Planning Algorithm on Grid Map
Author: Senior Software Engineer
Environment: Python 3.9.5, PyCharm
Description:
    Implements A* search algorithm for global path planning on a 2D grid map.
    Includes multiple heuristic functions and expansion strategies.
"""

import heapq
import math
from typing import List, Tuple, Optional, Callable, Dict
import matplotlib.pyplot as plt
import numpy as np


class Node:
    """
    节点类：用于表示栅格地图中的搜索节点
    """

    def __init__(self, position: Tuple[int, int], parent: Optional['Node'] = None):
        self.position = position  # (row, col) 坐标
        self.parent = parent  # 父节点，用于回溯路径
        self.g = 0.0  # 起点到当前节点的实际代价
        self.h = 0.0  # 当前节点到目标点的启发式估计代价
        self.f = 0.0  # 总代价 f = g + h

    def __lt__(self, other: 'Node'):
        """定义小于运算符，用于优先队列(heapq)的比较"""
        return self.f < other.f

    def __eq__(self, other: object):
        """定义等于运算符，用于判断是否为同一节点"""
        if not isinstance(other, Node):
            return False
        return self.position == other.position

    def __hash__(self):
        """定义哈希值，使其可以作为字典的键或放入集合中"""
        return hash(self.position)


class GridMap:
    """
    栅格地图类：负责地图的表示、验证与基础操作
    """

    def __init__(self, grid: List[List[int]]):
        """
        初始化地图
        :param grid: 二维数组，0表示可通行，1表示障碍物
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """检查坐标是否在地图范围内且不是障碍物"""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0


class Heuristics:
    """
    启发式函数集合：提供多种距离计算方式
    """

    @staticmethod
    def manhattan(current: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """曼哈顿距离：适用于4邻域拓展"""
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    @staticmethod
    def euclidean(current: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """欧几里得距离：直线距离"""
        return math.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)

    @staticmethod
    def chebyshev(current: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """切比雪夫距离：适用于8邻域拓展"""
        return max(abs(current[0] - goal[0]), abs(current[1] - goal[1]))

    @staticmethod
    def diagonal(current: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """对角线距离 (Octile Distance)：考虑了对角线移动的代价(根号2)"""
        dx = abs(current[0] - goal[0])
        dy = abs(current[1] - goal[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


class ExpansionStrategies:
    """
    拓展策略集合：定义节点的邻居生成方式
    """

    @staticmethod
    def four_connected(pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """
        4邻域拓展 (上, 下, 左, 右)
        :return: [(邻居坐标, 移动代价), ...]
        """
        r, c = pos
        neighbors = [
            ((r - 1, c), 1.0), ((r + 1, c), 1.0),
            ((r, c - 1), 1.0), ((r, c + 1), 1.0)
        ]
        return neighbors

    @staticmethod
    def eight_connected(pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """
        8邻域拓展 (包含对角线方向)
        :return: [(邻居坐标, 移动代价), ...]
        """
        r, c = pos
        sqrt2 = math.sqrt(2)
        neighbors = [
            ((r - 1, c), 1.0), ((r + 1, c), 1.0),
            ((r, c - 1), 1.0), ((r, c + 1), 1.0),
            ((r - 1, c - 1), sqrt2), ((r - 1, c + 1), sqrt2),
            ((r + 1, c - 1), sqrt2), ((r + 1, c + 1), sqrt2)
        ]
        return neighbors


class AStarPlanner:
    """
    A* 路径规划器核心类
    """

    def __init__(self, grid_map: GridMap,
                 heuristic_func: Callable[[Tuple[int, int], Tuple[int, int]], float] = Heuristics.manhattan,
                 expansion_func: Callable[
                     [Tuple[int, int]], List[Tuple[Tuple[int, int], float]]] = ExpansionStrategies.four_connected):
        """
        :param grid_map: 栅格地图对象
        :param heuristic_func: 启发式函数
        :param expansion_func: 拓展策略函数
        """
        self.map = grid_map
        self.heuristic = heuristic_func
        self.expansion = expansion_func
        self.searched_nodes: List[Tuple[int, int]] = []  # 记录搜索过的节点，用于可视化

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        执行 A* 路径搜索
        :param start: 起点坐标 (row, col)
        :param goal: 终点坐标 (row, col)
        :return: 路径坐标序列，若找不到路径则返回 None
        """
        # 1. 输入异常处理与校验
        if not self.map.is_valid(start):
            raise ValueError(f"起点 {start} 无效（越界或为障碍物）")
        if not self.map.is_valid(goal):
            raise ValueError(f"终点 {goal} 无效（越界或为障碍物）")

        if start == goal:
            return [start]

        self.searched_nodes.clear()

        # 2. 初始化起点
        start_node = Node(position=start)
        start_node.g = 0.0
        start_node.h = self.heuristic(start, goal)
        start_node.f = start_node.g + start_node.h

        # 3. 初始化 Open List (优先队列) 和 Closed List (集合，O(1)查找)
        open_list: List[Node] = []
        heapq.heappush(open_list, start_node)

        closed_set: set = set()

        # 用于快速查找 open list 中已存在的节点及其代价
        open_dict: Dict[Tuple[int, int], float] = {start: start_node.f}

        # 4. 核心搜索循环
        while open_list:
            # 取出 f 值最小的节点
            current_node = heapq.heappop(open_list)
            current_pos = current_node.position

            # 如果该节点已在 closed 集中，跳过（处理堆中的冗余节点）
            if current_pos in closed_set:
                continue

            # 将当前节点加入 closed 集
            closed_set.add(current_pos)
            self.searched_nodes.append(current_pos)

            # 5. 目标检测
            if current_pos == goal:
                return self._reconstruct_path(current_node)

            # 6. 拓展邻居节点
            for neighbor_pos, move_cost in self.expansion(current_pos):
                # 检查邻居是否有效且未被访问过
                if not self.map.is_valid(neighbor_pos) or neighbor_pos in closed_set:
                    continue

                # 计算新的 g 值
                tentative_g = current_node.g + move_cost

                # 创建邻居节点
                neighbor_node = Node(position=neighbor_pos, parent=current_node)
                neighbor_node.g = tentative_g
                neighbor_node.h = self.heuristic(neighbor_pos, goal)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                # 如果邻居不在 open_dict 中，或者找到了更优的路径
                if neighbor_pos not in open_dict or tentative_g < open_dict[neighbor_pos]:
                    open_dict[neighbor_pos] = tentative_g
                    heapq.heappush(open_list, neighbor_node)

        # 7. Open List 为空仍未找到目标，说明无可行路径
        print("警告：无法找到从起点到终点的可行路径！")
        return None

    def _reconstruct_path(self, current_node: Node) -> List[Tuple[int, int]]:
        """回溯生成路径"""
        path = []
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        return path[::-1]  # 反转列表，使其从起点到终点


def visualize_path(grid_map: GridMap, start: Tuple[int, int], goal: Tuple[int, int],
                   path: Optional[List[Tuple[int, int]]], searched_nodes: List[Tuple[int, int]], title: str):
    """
    使用 Matplotlib 可视化地图、搜索区域和最终路径
    """
    # 将网格转换为 numpy 数组以便绘图
    grid_np = np.array(grid_map.grid)

    plt.figure(figsize=(8, 8))
    # 绘制地图背景 (白色为0，黑色为1)
    cmap = plt.cm.gray_r
    plt.imshow(grid_np, cmap=cmap, origin='upper')

    # 绘制搜索过的区域 (浅蓝色)
    if searched_nodes:
        searched_rows, searched_cols = zip(*searched_nodes)
        plt.scatter(searched_cols, searched_rows, color='lightblue', s=20, zorder=2, label='Searched Area')

    # 绘制路径 (红色线)
    if path:
        path_rows, path_cols = zip(*path)
        plt.plot(path_cols, path_rows, color='red', linewidth=2, zorder=3, label='Path')

    # 绘制起点 (绿色) 和 终点 (红色)
    plt.scatter(start[1], start[0], color='green', s=100, marker='o', zorder=4, label='Start')
    plt.scatter(goal[1], goal[0], color='red', s=100, marker='*', zorder=4, label='Goal')

    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：处理输入输出，演示不同启发式函数和拓展策略的组合
    """
    # 1. 输入：定义栅格地图 (0: 可通行, 1: 障碍物)
    grid_data = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    grid_map = GridMap(grid_data)

    # 2. 输入：定义起点和终点 (row, col)
    start_point = (0, 0)
    goal_point = (9, 9)

    # 配置不同的测试用例：(启发式函数, 拓展策略, 标题)
    test_configs = [
        (Heuristics.manhattan, ExpansionStrategies.four_connected, "A* with Manhattan & 4-Connected"),
        (Heuristics.diagonal, ExpansionStrategies.eight_connected, "A* with Diagonal & 8-Connected"),
        (Heuristics.euclidean, ExpansionStrategies.eight_connected, "A* with Euclidean & 8-Connected")
    ]

    for heu_func, exp_func, title in test_configs:
        print(f"\n--- 正在运行: {title} ---")

        # 实例化规划器
        planner = AStarPlanner(grid_map, heuristic_func=heu_func, expansion_func=exp_func)

        try:
            # 3. 执行算法
            path = planner.plan(start_point, goal_point)

            # 4. 输出结果
            if path:
                print(f"成功找到路径! 路径长度(节点数): {len(path)}")
                print(f"路径坐标序列: {path}")
            else:
                print("未找到可行路径。")

            # 5. 可视化
            visualize_path(grid_map, start_point, goal_point, path, planner.searched_nodes, title)

        except ValueError as e:
            print(f"输入错误: {e}")
        except Exception as e:
            print(f"发生未知异常: {e}")


if __name__ == "__main__":
    # 确保安装了依赖库: pip install matplotlib numpy
    main()