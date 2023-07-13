# @Time : 2023/5/14 12:04 
# @Author : Yinquan Wang<19114012@bjtu.edu.cn>
# @File : Driver.py 
# @Function:
import random
import numpy as np
from collections import deque


class Drivers:
    def __init__(self, driver_path):
        """
        Initialize the Driver object, loading a given number of drivers from a file, and initializing their attributes.

        Parameters:
            driver_path: String, the path of the file that contains the driver data.

        Attributes:
            self.drivers_list: List, the loaded drivers.
            self.array: numpy.ndarray, a structured data type to store various attributes of drivers, with the following columns:
                0: Unique driver ID.
                1: Longitude of driver's current position.
                2: Latitude of driver's current position.
                3: The time the driver goes online.
                4: The time the driver goes offline.
                5: The index of the passenger currently matched with the driver.
                6: Current status of the driver (-1: offline, 0: idle, 1: pickup, 2: delivery, 3: reposition, 4: offline).
                7: Cumulative idle time of the driver.
                8: Total income of the driver.
                9: Number of orders the driver has accepted.
                10: Type of driver (0: full-time, 1: part-time).
                11: Node ID of the driver's current position in real-time road network.
                12: Node ID of the driver's current destination in real-time road network.
                13: ID of the road segment where the driver is currently located.
            pickup_deque, delivery_deque, reposition_deque: deque, deques to store pickup, delivery,
            and reposition trajectories for each driver respectively.
            pickup_route_deque, delivery_route_deque, reposition_route_deque: deque, deques to store routes for pickup,
            delivery, and reposition for each driver respectively.
        """
        self.drivers_list = random.sample(list(np.load(driver_path, allow_pickle=True)), k=10000)
        # self.drivers_list = list(np.load(driver_path, allow_pickle=True))

        self.array = np.zeros(shape=(len(self.drivers_list), 14))
        for driver_index, driver in enumerate(self.drivers_list):
            self.array[driver_index, 0] = int(driver_index)  # 司机的唯一ID
            self.array[driver_index, 1] = float(driver['initial_position_x'])  # 司机位置的经度
            self.array[driver_index, 2] = float(driver['initial_position_y'])  # 司机位置的纬度
            self.array[driver_index, 3] = int(driver['online_time'])  # 司机上线时间
            self.array[driver_index, 4] = int(driver['offline_time'])  # 司机下线时间
            self.array[driver_index, 5] = -1  # 司机当前匹配的乘客的index
            self.array[driver_index, 6] = -1  # 司机状态（空闲、接客、载客、下线）
            self.array[driver_index, 7] = 0  # 司机累计空闲时间
            self.array[driver_index, 8] = 0  # 司机工作收入
            self.array[driver_index, 9] = 0  # 司机接单数量
            self.array[driver_index, 10] = np.random.randint(low=1, high=2)  # 司机类型（0：雇佣制，1：兼职制）
        self.array[:, 11] = -1  # 实时路网上司机的当前位置节点ID
        self.array[:, 12] = -1  # 司机当前行驶的目的地节点ID
        self.array[:, 13] = -1  # 为司机所在的路段的编号

        self.pickup_deque = [deque() for i in range(self.array.shape[0])]  # pick up trajectory
        self.delivery_deque = [deque() for i in range(self.array.shape[0])]  # delivery trajectory
        self.reposition_deque = [deque() for i in range(self.array.shape[0])]  # reposition trajectory
        self.pickup_route_deque = [deque() for i in range(self.array.shape[0])]
        self.delivery_route_deque = [deque() for i in range(self.array.shape[0])]
        self.reposition_route_deque = [deque() for i in range(self.array.shape[0])]

    def update_nodes(self, origin_road):
        """
        Updates the node information for all the drivers or passengers based on their current positions.

        Parameters:
            origin_road (RoadNetwork object): An instance of a road network class that provides the node information.
        """
        self.array[:, 11] = origin_road.pandana_net.get_node_ids(self.array[:, 1], self.array[:, 2])
        self.array[:, 11] = self.array[:, 11].astype(int)

    def begin_work(self, env_time):
        """
        Change the state of the drivers who start their work shift at the current time to 'online'.

        Parameters:
            env_time (int): current time in the environment.
        """
        online_driver = np.where((self.array[:, 6] == -1) & (self.array[:, 3] == env_time))[0]
        self.array[online_driver, 6] = 0
        return online_driver

    def position(self, vehicle_supply):
        """
        Get the current position of specific drivers or all drivers.

        Parameters:
            vehicle_supply (numpy.ndarray or str): the indices of the drivers or 'all' to include all drivers.
        Returns:
            The current positions of the specified drivers.
        """
        if isinstance(vehicle_supply, str) and vehicle_supply == 'all':
            return self.array[:, [1, 2]]
        return self.array[vehicle_supply, :][:, [1, 2]]

    def nodes(self, vehicle_supply):
        """
        Get the current node ID of specific drivers or all drivers in the road network.

        Parameters:
            vehicle_supply: numpy.ndarray or 'all', the indices of the drivers or 'all' to include all drivers.
        Returns:
            The current node IDs of the specified drivers.
        """
        if isinstance(vehicle_supply, str) and vehicle_supply == 'all':
            return self.array[:, 11]
        return self.array[vehicle_supply, 11]

    def destinations(self, vehicle_supply):
        """
        Get the current destination node ID of specific drivers or all drivers.

        Parameters:
            vehicle_supply: numpy.ndarray or 'all', the indices of the drivers or 'all' to include all drivers.
        Returns:
            The current destination node IDs of the specified drivers.
        """
        if isinstance(vehicle_supply, str) and vehicle_supply == 'all':
            return self.array[:, 12]
        return self.array[vehicle_supply, 12]

    def routes(self, vehicle_supply):
        """
        Get the current route ID of specific drivers or all drivers.

        Parameters:
            vehicle_supply: numpy.ndarray or 'all', the indices of the drivers or 'all' to include all drivers.
        Returns:
            The current route IDs of the specified drivers.
        """
        if isinstance(vehicle_supply, str) and vehicle_supply == 'all':
            return self.array[:, 13]
        return self.array[vehicle_supply, 13]

    @property
    def idle(self):
        """
        Get the indices of drivers who are currently idle.
        Returns:
            The indices of idle drivers.
        """
        return np.where(self.array[:, 6] == 0)[0]

    @property
    def pick(self):
        """
        Get the indices of drivers who are currently picking up a passenger.
        Returns:
            The indices of drivers in pick-up mode.
        """
        return np.where(self.array[:, 6] == 1)[0]

    @property
    def delivery(self):
        """
        Get the indices of drivers who are currently delivering a passenger.
        Returns:
            The indices of drivers in delivery mode.
        """
        return np.where(self.array[:, 6] == 2)[0]

    @property
    def reposition(self):
        """
        Get the indices of drivers who are currently in reposition mode.
        Returns:
            The indices of drivers in reposition mode.
        """
        return np.where(self.array[:, 6] == 3)[0]

    @property
    def operating(self):  # 获取当前系统中的运营的车辆
        """
        Get the indices of drivers who are currently operating (either idle, picking, delivering, or in reposition).
        Returns:
            The indices of operating drivers.
        """
        return np.where(np.isin(self.array[:, 6], [0, 1, 2, 3]))[0]

    @property
    def operating_on_road(self):
        """
        Get the indices of drivers who are currently operating on the road (either picking, delivering, or in reposition).
        Returns:
            The indices of drivers operating on the road.
        """
        return np.where(np.isin(self.array[:, 6], [1, 2, 3]))[0]

    def will_offline(self, env_time):
        """
        Get the indices of drivers who will go offline at the current time.

        Parameters:
            env_time: int, current time in the environment.
        Returns:
            The indices of drivers who will go offline.
        """
        return np.where((self.array[:, 4] <= env_time) & (self.array[:, 6] == 0))

    def info(self, vehicle_supply, column_index=None):
        """
        Get the information of specific drivers.

        Parameters:
            vehicle_supply: numpy.ndarray, the indices of the drivers.
            column_index: int or None, the column index of the information to retrieve. If None, all information will be
            retrieved.
        Returns:
            The requested information of the specified drivers.
        """
        if column_index is None:
            return self.array[vehicle_supply, :]
        return self.array[vehicle_supply, column_index]

    def assign(self, vehicle_supply, column_index, value):
        """
        Assign a new value to a specific attribute of certain drivers.

        Parameters:
            vehicle_supply: numpy.ndarray, the indices of the drivers.
            column_index: int, the column index of the attribute to assign the new value.
            value: the new value to be assigned.
        """
        self.array[vehicle_supply, column_index] = value

    def matched_passenger(self, vehicle_supply):
        """
        Get the index of the passenger currently matched with specific drivers.

        Parameters:
            vehicle_supply: numpy.ndarray, the indices of the drivers.
        Returns:
            The index of the matched passenger.
        """
        return self.array[vehicle_supply, 5].astype(int)

    def update_pickup_position(self):
        """
        Update the positions of drivers who are currently picking up passengers.

        Returns:
            List of drivers who have successfully picked up their passengers.
        """
        pickup_drivers = np.where(self.array[:, 6] == 1)[0].tolist()  # 筛选得到处于接客过程中的司机
        to_delivery = []
        for driver_index in pickup_drivers:
            try:
                next_position = self.pickup_deque[driver_index].popleft()
                next_route = self.pickup_route_deque[driver_index].popleft()
            except IndexError:
                to_delivery.append(self.array[driver_index, 5].astype(int))
                next_position = self.delivery_deque[driver_index].popleft()
                next_route = self.delivery_route_deque[driver_index].popleft()
                self.pickup_deque[driver_index] = deque()  # 司机已经接了乘客，接客deque更改为空deque
                self.pickup_route_deque[driver_index] = deque()
                self.array[driver_index, 6] = 2  # 司机状态更改为载客
            self.array[driver_index, 13] = next_route
            self.array[driver_index, 1] = next_position[0]  # 更新车辆位置
            self.array[driver_index, 2] = next_position[1]  # 更新车辆位置
        return to_delivery

    def update_delivery_position(self):
        """
        Update the positions of drivers who are currently delivering passengers.

        Returns:
            List of drivers who have successfully delivered their passengers.
        """
        delivery_drivers = np.where(self.array[:, 6] == 2)[0].tolist()  # 筛选得到处于载客过程中的司机
        to_finish = []
        for driver_index in delivery_drivers:
            try:
                next_position = self.delivery_deque[driver_index].popleft()  # 获取司机的当前位置
                next_route = self.delivery_route_deque[driver_index].popleft()
                self.array[driver_index, 1] = next_position[0]  # 更新司机的坐标
                self.array[driver_index, 2] = next_position[1]  # 更新司机的坐标
                self.array[driver_index, 13] = next_route
            except IndexError:
                to_finish.append(self.array[driver_index, 5].astype(int))
                self.delivery_deque[driver_index] = deque()  # 将那些已完成当前订单的司机的载客轨迹更改为空
                self.delivery_route_deque[driver_index] = deque()
                self.array[driver_index, 5] = -1  # 司机所匹配的乘客index为空
                self.array[driver_index, 6] = 0  # 将完成当前订单的司机的状态更改为空闲
                # 将司机的节点位置更新为初始路网上的乘客到达节点的index
                self.array[driver_index, 12] = -1
        return to_finish

    def update_reposition_position(self):
        reposition_drivers = np.where(self.array[:, 6] == 3)[0].tolist()
        for driver_index in reposition_drivers:
            try:
                next_position = self.reposition_deque[driver_index].popleft()
                next_route = self.reposition_route_deque[driver_index].popleft()
                self.array[driver_index, 1] = next_position[0]
                self.array[driver_index, 2] = next_position[1]
                self.array[driver_index, 13] = next_route
            except IndexError:
                self.reposition_deque[driver_index] = deque()
                self.reposition_route_deque[driver_index] = deque()
                self.array[driver_index, 6] = 0

    def update_pick_trajectories_and_route(self, vehicle_supply, new_trajectories, new_route):
        """
        Update the pick-up trajectories and routes of certain drivers.

        Parameters:
            vehicle_supply: numpy.ndarray, the indices of the drivers.
            new_trajectories: list, new pick-up trajectories for the drivers.
            new_route: list, new pick-up routes for the drivers.
        """
        for i, driver_name in enumerate(vehicle_supply):
            self.pickup_deque[driver_name] = deque(new_trajectories[i])
            self.pickup_route_deque[driver_name] = deque(new_route[i])

    def update_delivery_trajectories_and_route(self, vehicle_supply, new_trajectories, new_route):  # 更新载客过程的轨迹
        """
        Update the delivery trajectories and routes of certain drivers.

        Parameters:
            vehicle_supply: numpy.ndarray, the indices of the drivers.
            new_trajectories: list, new delivery trajectories for the drivers.
            new_route: list, new delivery routes for the drivers.
        """
        for i, driver_name in enumerate(vehicle_supply):
            self.delivery_deque[driver_name] = deque(new_trajectories[i])
            self.delivery_route_deque[driver_name] = deque(new_route[i])

    def update_reposition_trajectories_route(self, vehicle_supply, new_trajectories, new_route):  # 更新调度过程的轨迹
        """
        Update the reposition trajectories and routes of certain drivers.

        Parameters:
            vehicle_supply: numpy.ndarray, the indices of the drivers.
            new_trajectories: list, new reposition trajectories for the drivers.
            new_route: list, new reposition routes for the drivers.
        """
        for i, driver_name in enumerate(vehicle_supply):
            self.reposition_deque[driver_name] = deque(new_trajectories[i])
            self.reposition_route_deque[driver_name] = deque(new_route[i])

    def store_info(self):
        """
        Store the current information of the online vehicles.

        Returns:
            List of lists, each inner list contains the ID, position, status, income, and the matched passenger index of
            an online vehicle.
        """
        online_vehicle = self.operating()
        store_trajectory = self.array[online_vehicle, :][:, [0, 1, 2, 6, 8, 5]].tolist()
        return store_trajectory
