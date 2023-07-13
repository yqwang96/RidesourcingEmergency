# @Time : 2023/5/14 12:04 
# @Author : Yinquan Wang<19114012@bjtu.edu.cn>
# @File : Passenger.py 
# @Function:
import random
import numpy as np
from collections import deque


class Passengers:
    def __init__(self, passenger_path: str, ):
        """
        Initializes an instance of the Passengers class.

        Parameters:
            passenger_path (str): The path to the file containing passenger information.

        self.array (np.ndarray): Each row corresponds to a passenger, and columns have the following meanings:
            0: Unique passenger ID.
            1: Longitude of passenger's origin.
            2: Latitude of passenger's origin.
            3: Longitude of passenger's destination.
            4: Latitude of passenger's destination.
            5: Passenger's departure time.
            6: Estimated travel distance (not used).
            7: Estimated travel duration (not used).
            8: Estimated travel price (not used).
            9: Passenger's waiting time.
            10: Passenger's status.
            11: Unique ID of the driver matched with the passenger.
            12: Distance to the matched passenger.
            13: Time to pick up the passenger.
            14: ID of the passenger's departure node in the real-time network.
            15: ID of the passenger's arrival node in the real-time network.
            16: Manhattan distance from the passenger's location to the nearest node.
        """
        self.passengers_list = random.sample(list(np.load(passenger_path, allow_pickle=True)), k=100000)
        # self.passengers_list = list(np.load(passenger_path, allow_pickle=True))

        self.array = np.zeros(shape=(len(self.passengers_list), 17))
        for passenger_index, passenger in enumerate(self.passengers_list):
            self.array[passenger_index, 0] = int(passenger_index)  # 乘客唯一ID
            self.array[passenger_index, 1] = float(passenger['origin_x'])  # 乘客出发点经度
            self.array[passenger_index, 2] = float(passenger['origin_y'])  # 乘客出发点纬度
            self.array[passenger_index, 3] = float(passenger['destination_x'])  # 乘客到达点经度
            self.array[passenger_index, 4] = float(passenger['destination_y'])  # 乘客到达点纬度
            self.array[passenger_index, 5] = int(passenger['travel_time'])  # 乘客出发时间
            # 6为预期出行里程, 7为预期出行时长, 8为预计出行费用,
            self.array[passenger_index, 9] = 0  # 乘客的等待时间
            self.array[passenger_index, 10] = 0  # 乘客的状态
            self.array[passenger_index, 11] = -1  # 乘客匹配到的司机的唯一ID
            self.array[passenger_index, 12] = -1  # 乘客匹配距离
            self.array[passenger_index, 13] = -1  # 乘客接客时间
        self.array[:, 14] = -1  # 实时路网上乘客出发节点的ID
        self.array[:, 15] = -1  # 实时路网上乘客到达节点的ID
        self.array[:, 16] = -1  # 乘客位置到最近的节点的曼哈顿走行距离
        # 9为实时出行里程, 10为实时出行时长, 11为实际出行费用

        self.walk_deque = [deque() for i in range(self.array.shape[0])]

    def update_dep_arr_nodes(self, origin_road):
        """
        Updates the node information for all the passengers based on their departure and arrival positions.

        Parameters:
            origin_road (RoadNetwork object): An instance of a road network class that provides the node information.
        """
        self.array[:, 14] = origin_road.pandana_net.get_node_ids(self.array[:, 1], self.array[:, 2])
        self.array[:, 15] = origin_road.pandana_net.get_node_ids(self.array[:, 3], self.array[:, 4])
        self.array[:, 14] = self.array[:, 14].astype(int)
        self.array[:, 15] = self.array[:, 15].astype(int)

    @property
    def num(self):
        """
        Returns the total number of passengers.

        Returns:
            int: The total number of passengers.
        """
        return self.array.shape[0]

    def dep_nodes(self, travel_demand):
        """
        Returns the IDs of departure nodes for specified passengers.

        Parameters:
            travel_demand (str, int or list): The indices of passengers.
            If 'all', return departure nodes for all passengers.

        Returns:
            np.ndarray: The IDs of departure nodes for specified passengers.
        """
        if isinstance(travel_demand, str) and travel_demand == 'all':
            return self.array[:, 14]
        return self.array[travel_demand, 14]

    def arr_nodes(self, travel_demand):
        """
        Returns the IDs of arrival nodes for specified passengers.

        Args:
            travel_demand (str, int or list): The indices of passengers.
            If 'all', return arrival nodes for all passengers.

        Returns:
            np.ndarray: The IDs of arrival nodes for specified passengers.
        """
        if isinstance(travel_demand, str) and travel_demand == 'all':
            return self.array[:, 15]
        return self.array[travel_demand, 15]

    def dep_coords(self, travel_demand):
        """
        Returns the departure coordinates for specified passengers.

        Args:
            travel_demand (str, int or list): The indices of passengers.
            If 'all', return departure coordinates for all passengers.

        Returns:
            np.ndarray: The departure coordinates for specified passengers.
        """
        if isinstance(travel_demand, str) and travel_demand == 'all':
            return self.array[:, 1:3]
        return self.array[travel_demand, 1:3]

    def arr_coords(self, travel_demand):
        """
        Returns the arrival coordinates for specified passengers.

        Args:
            travel_demand (str, int or list): The indices of passengers.
            If 'all', return arrival coordinates for all passengers.

        Returns:
            np.ndarray: The arrival coordinates for specified passengers.
        """
        if isinstance(travel_demand, str) and travel_demand == 'all':
            return self.array[:, [3, 4]]
        return self.array[travel_demand, :][:, [3, 4]]

    @property
    def need_onboard(self):  # 得到当前时刻的需要前往最近节点的乘客
        """
        Returns the indices of passengers who need to move towards the nearest node at the current time.

        Returns:
            np.ndarray: The indices of passengers who need to move towards the nearest node.
        """
        return np.where(self.array[:, 10] == 1)[0]

    @property
    def during_onboard(self):  # 得到当前时刻的正在前往最近节点的乘客
        """
        Returns the indices of passengers who are moving towards the nearest node at the current time.

        Returns:
            np.ndarray: The indices of passengers who are moving towards the nearest node.
        """
        return np.where(self.array[:, 10] == 2)[0]

    @property
    def waiting_service(self):
        """
        Returns the indices of passengers who are waiting for a match.

        Returns:
            np.ndarray: The indices of passengers who are waiting for a match.
        """
        return np.where(self.array[:, 10] == 3)[0]  # 乘客状态为等待匹配

    @property
    def during_pick(self):
        """
        Returns the indices of passengers who are currently being picked up.

        Returns:
            np.ndarray: The indices of passengers who are currently being picked up.
        """
        return np.where(self.array[:, 10] == 4)[0]

    @property
    def during_delivery(self):
        """
        Returns the indices of passengers who are currently being delivered.

        Returns:
            np.ndarray: The indices of passengers who are currently being delivered.
        """
        return np.where(self.array[:, 10] == 5)[0]

    @property
    def completed(self):
        """
        Returns the indices of passengers who have completed their journey.

        Returns:
            np.ndarray: The indices of passengers who have completed their journey.
        """
        return np.where(self.array[:, 10] == 6)[0]

    @property
    def canceled(self):
        """
        Returns the indices of passengers whose journey has been canceled.

        Returns:
            np.ndarray: The indices of passengers whose journey has been canceled.
        """
        return np.where(self.array[:, 10] == 7)[0]

    @property
    def wait_pick(self):
        """
        Returns the indices of passengers who are either waiting matching or being picked up.

        Returns:
            np.ndarray: The indices of passengers who are either waiting, moving towards the nearest node, or being picked up.
        """
        return np.where(np.isin(self.array[:, 10], [3, 4]))[0]

    @property
    def walk_wait_pick(self):
        """
        Returns the indices of passengers who are either waiting, moving towards the nearest node, or being picked up.

        Returns:
            np.ndarray: The indices of passengers who are either waiting, moving towards the nearest node, or being picked up.
        """
        return np.where(np.isin(self.array[:, 10], [1, 2, 3, 4]))[0]

    @property
    def unactivated(self):  # 获取状态为未激活的乘客
        """
        Returns the indices of passengers who are not yet activated.

        Returns:
            np.ndarray: The indices of passengers who are not yet activated.
        """
        return np.where(self.array[:, 10] == 0)[0]

    def control_cancel(self, travel_demand):  # 更改限制区域内等待匹配、前往上车点和等待接客的乘客的状态
        """
        Changes the status of passengers within a controlled area who are waiting for a match, moving towards the pick-up
        point, or waiting for a pick-up.

        Args:
            travel_demand (np.ndarray): The indices of passengers within a controlled area.
        """
        self.array[travel_demand, 10] = 8

    def match_state_update(self, travel_demand, their_driver_name):
        """
        Updates the state of matched passengers to 'being picked up' and assigns the index of their matched driver.

        Args:
            travel_demand (np.ndarray): The indices of passengers who have been matched with a driver.
            their_driver_name (list): The indices of the drivers matched with the passengers.
        """
        self.array[travel_demand, 10] = 4  # 乘客状态为接客中
        self.array[travel_demand, 11] = np.array(their_driver_name)  # 更新乘客所匹配到的司机的index

    def unmatch_state_update(self, travel_demand):
        """
        Increments the waiting time of passengers who have not been matched with a driver.

        Args:
            travel_demand (np.ndarray): The indices of passengers who have not been matched with a driver.
        """
        self.array[travel_demand, 9] += 1  # 没有被匹配的乘客的等待时间+1

    def update_onboard_position(self):  # 更新上车乘客的状态
        """
        Updates the status of passengers who are on board.
        """
        onboard_passenger = self.during_onboard.tolist()
        for passenger_index in onboard_passenger:
            try:
                next_position = self.walk_deque[passenger_index].popleft()
                self.array[passenger_index, 1] = next_position[0]
                self.array[passenger_index, 2] = next_position[1]
            except IndexError:
                self.array[passenger_index, 10] = 3

    def activate_passengers(self, env_time):  # 激活那些在当前时间窗出行的乘客
        """
        Activates passengers who are scheduled to travel at the current environment time.

        Args:
            env_time (int): The current environment time.
        """
        travel_passenger = np.where((self.array[:, 10] == 0) &
                                    (self.array[:, 5] == env_time))[0]
        self.array[travel_passenger, 10] = 1
        return travel_passenger

    def update_canceled(self):
        """
        Updates the status of passengers who have been waiting for 10 or more time units to 'Canceled'.

        Returns:
            int: The number of passengers whose status has been updated to 'Canceled'.
        """
        # 根据等待时间阈值，筛选取消出行请求的乘客
        cancel_passengers = np.where(self.array[:, 9] >= 10)[0]
        self.array[cancel_passengers, 10] = 7  # 将这部分乘客的状态更改为‘取消’
        return len(cancel_passengers)

    def info(self, travel_demand, column_index=None):
        """
        Returns information about specific passengers. If no specific column is provided, returns all information for
        those passengers.

        Args:
            travel_demand (np.ndarray): The indices of passengers.
            column_index (int, optional): The specific column index of interest.

        Returns:
            np.array: The requested information about the passengers.
        """
        if column_index is None:
            return self.array[travel_demand, :]
        return self.array[travel_demand, column_index]

    def assign(self, travel_demand, column_index, value):
        """
        Assigns a value to a specific column for specific passengers.

        Args:
            travel_demand (np.ndarray): The indices of passengers.
            column_index (int): The column index to assign the value to.
            value (numeric or np.ndarray): The value to be assigned.
        """
        self.array[travel_demand, column_index] = value

    def pick2serve(self, passenger_name):
        """
        Updates the status of a passenger from 'Picked up' to 'Being served'.

        Args:
            passenger_name (int or list): The index of the passenger.
        """
        self.array[passenger_name, 10] = 5

    def serve2finish(self, passenger_name):
        """
        Updates the status of a passenger from 'Being served' to 'Finished'.

        Args:
            passenger_name (int or list): The index of the passenger.
        """
        self.array[passenger_name, 10] = 6  # 更新乘客状态为‘完成’

    def update_onboard_trajectories(self, travel_demand, walk_trajectories):
        """
        Updates the walking trajectories of specific passengers.

        Parameters:
            travel_demand (np.ndarray): The indices of passengers.
            walk_trajectories (list of lists): The new walking trajectories for the passengers.
        """
        for i, passenger_name in enumerate(travel_demand):
            self.walk_deque[passenger_name] = deque(walk_trajectories[i])

