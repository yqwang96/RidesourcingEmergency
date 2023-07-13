# -*- coding: utf-8
# @Time : 2023/6/18 16:52
# @Author: Yinquan WANG
# @Email : 19114012@bjtu.edu.cn
# @File : Platform.py
# @Project : 网约出行仿真
import numpy as np
from lapsolver import solve_dense
from static.Passenger import Passengers
from static.Driver import Drivers
from static.Util import *


class Platform:
    def __init__(
            self, name: str, operate_type: str, passenger_path: str, driver_path: str, base_fare: float,
            unit_time_fare: float, unit_mileage_fare: float, commission_ratio: float, max_match_radius: float,
            max_pick_times: float, dispatch_algorithm: str, reposition_algorithm: str, nearest_limitation: int,
            simulation_type: str, dynamic_impedance: bool
    ):
        """
        Initialize the simulation environment with given parameters.

        Parameters:
            name (str): The name of the simulation environment.
            operate_type (str): The operation type, it can be 'heavy', 'light', or 'mix'.
            passenger_path (str): The path to the passenger data.
            driver_path (str): The path to the driver data.
            base_fare (float): The base fare for the ride (CNY),
            unit_time_fare (float): The fare per unit time (CNY/min).
            unit_mileage_fare (float): The fare per unit mileage (CNY/km).
            commission_ratio (float): The commission ratio, range 0.0~1.0.
            max_match_radius (float): The maximum match radius for dispatching, (km).
            max_pick_times (float): The maximum time for a driver to pick up a passenger, (min).
            dispatch_algorithm (str): The dispatch algorithm to be used.
            reposition_algorithm (str): The reposition algorithm to be used.
            nearest_limitation (int): The limit for nearest match.
            simulation_type (str): The type of simulation to be run.
            dynamic_impedance (bool): Whether to consider dynamic impedance or not.
        """
        self.name = name
        self.operate_type = operate_type

        self.passenger_path = passenger_path
        self.driver_path = driver_path

        self.passengers = Passengers(self.passenger_path)
        self.drivers = Drivers(self.driver_path)

        self.base_fare = base_fare
        self.unit_time_fare = unit_time_fare
        self.unit_mileage_fare = unit_mileage_fare

        if self.operate_type == 'heavy':  # 运营类型有三种：自有车队、非自有车队、混合模式
            self.commission_ratio = 1
        elif self.operate_type == 'light':
            self.commission_ratio = commission_ratio
        elif self.operate_type == 'mix':
            self.commission_ratio = commission_ratio

        self.max_match_radius = max_match_radius
        self.max_pick_times = max_pick_times

        self.nearest_num = nearest_limitation

        self.dispatch_algorithm = dispatch_algorithm
        self.reposition_algorithm = reposition_algorithm

        self.simulation_type = simulation_type
        self.dynamic_impedance = dynamic_impedance

        # 评估指标(仿真完成后评估)
        self.platform_revenue = 0          # 平台收入
        self.completed_requests = 0        # 完成订单数
        self.order_response_rate = 0       # 订单响应率
        self.cancel_requests = 0           # 取消请求数
        self.cancel_rate = 0               # 订单取消率
        self.avg_match_distance = 0        # 平均匹配距离
        self.avg_pick_times = 0            # 平均接客时长
        self.avg_waiting_times = 0         # 平均等待时长
        self.avg_driver_income = 0         # 平均司机收入
        self.avg_driver_order = 0          # 平均司机订单数
        self.avg_driver_idle_duration = 0  # 平均司机空闲时长

        # 监测指标(仿真进行中监测)
        self.waiting_passenger_num = 0             # 等待乘客数
        self.on_board_passenger_num = 0            # 前往上车点乘客数
        self.during_pick_passenger_num = 0         # 正在等待接客乘客数
        self.during_delivery_passenger_num = 0     # 正在前往目的地的乘客数

        self.idle_driver_num = 0                   # 空闲司机数
        self.pick_driver_num = 0                   # 接客司机数
        self.delivery_driver_num = 0               # 载客司机数
        self.reposition_driver_num = 0             # 调度司机数

        self.step_platform_revenue = 0             # 单步平台收入
        self.step_completed_requests = 0           # 单步完成请求数
        self.step_order_response_rate = 0          # 单步订单响应率
        self.step_cancel_requests = 0              # 单步取消请求数
        self.step_cancel_rate = 0                  # 单步订单取消率
        self.step_match_distance = 0               # 单步平均匹配距离
        self.step_pick_times = 0                   # 单步平均接客时长
        self.step_waiting_times = 0                # 单步平均等待时长

        self.step_platform_revenue_list = []
        self.step_completed_requests_list = []
        self.step_order_response_rate_list = []
        self.step_cancel_requests_list = []
        self.step_cancel_rate_list = []
        self.step_match_distance_list = []
        self.step_pick_times_list = []
        self.step_waiting_times_list = []

    def update_passenger_travel_info(self, road):    # 更新乘客的预期出行时长、预期出行里程和预期出行费用
        """
        Update passenger's estimated travel mileage, duration, and price based on the current road condition.

        Parameters:
            road (object): The road network object.
        """
        self.update_estimated_travel_mileage(road)
        self.update_estimated_travel_duration(road)
        travel_demand = self.passengers.wait_pick
        self.update_estimated_travel_price(travel_demand)

    def update_estimated_travel_mileage(self, road):  # 更新预期出行里程
        """
        Update passenger's estimated travel mileage based on the current road condition.

        Parameters:
            road (RoadNetwork object): The road network object.
        """
        update_passenger = self.passengers.wait_pick
        dep_nodes = self.passengers.dep_nodes(update_passenger)
        arr_nodes = self.passengers.arr_nodes(update_passenger)
        distance = road.shortest_path_lengths(dep_nodes, arr_nodes, imp_name='length')
        self.passengers.array[update_passenger, 6] = distance

    def update_estimated_travel_duration(self, road):  # 更新预期出行时长
        """
        Update passenger's estimated travel duration based on the current road condition.

        Parameters:
            road (RoadNetwork object): The road network object.
        """
        update_passenger = self.passengers.wait_pick
        dep_nodes = self.passengers.dep_nodes(update_passenger)
        arr_nodes = self.passengers.arr_nodes(update_passenger)
        times = road.shortest_path_lengths(dep_nodes, arr_nodes, imp_name='times')
        self.passengers.array[update_passenger, 7] = times

    def update_estimated_travel_price(self, travel_demand):  # 这里可以引入其他定制化的价格计算方法
        """
        Update passenger's estimated travel price based on the current travel demand.
        Can introduce other customized pricing calculation methods here.

        Parameters:
            travel_demand (numpy.ndarray): A numpy array representing the travel demand.
        """
        self.passengers.array[travel_demand, 8] = self.base_fare + \
            self.unit_mileage_fare * self.passengers.array[travel_demand, 6] / 1000 +\
            self.unit_time_fare * self.passengers.array[travel_demand, 7]

    def waiting_passengers(self):
        """
        Get the passengers who are waiting for the service.

        Parameters:
        None

        Returns:
            numpy.ndarray: A numpy array representing the waiting passengers.
        """
        return self.passengers.waiting_service

    def idle_drivers(self):
        """
        Get the drivers who are currently idle.

        Parameters:
        None

        Returns:
            numpy.ndarray: A numpy array representing the idle drivers.
        """
        return self.drivers.idle

    def get_feasible_matching(self, travel_demand, vehicle_supply):
        """
        Generate feasible matchings between passengers and drivers.

        Parameters:
            travel_demand (np.ndarray): Travel demand, likely passenger IDs.
            vehicle_supply (np.ndarray): Vehicle supply, likely driver IDs.

        Returns:
            numpy.ndarray: Feasible matchings, each row is a matching pair.
        """
        passenger_info = self.passengers.info(travel_demand)
        driver_info = self.drivers.info(vehicle_supply)
        feasible_matching = np.hstack((np.repeat(driver_info, passenger_info.shape[0], 0),
                                       np.tile(passenger_info, (driver_info.shape[0], 1))))
        return feasible_matching

    def calculate_distance(self, feasible_matching, travel_demand, vehicle_supply, road):
        """
        Calculate the shortest path distance for each feasible matching.

        Parameters:
            feasible_matching (numpy.ndarray): Feasible matchings between drivers/passengers.
            travel_demand (numpy.ndarray): Travel demand.
            vehicle_supply (numpy.ndarray): Vehicle supply.
            road (RoadNetwork object): Road network object.

        Returns:
            np.ndarray: Shortest path distances, reshaped (vehicle_supply size, travel_demand size).
        """
        distances = road.shortest_path_lengths(
            feasible_matching[:, 11], feasible_matching[:, 28], imp_name='length'
        )
        distances_array = np.array(distances).reshape(vehicle_supply.shape[0], travel_demand.shape[0])
        return distances_array

    def calculate_pick_times(self, feasible_matching, travel_demand, vehicle_supply, road):
        """
        Calculate the shortest path pick times for each feasible matching.

        Parameters:
            feasible_matching (numpy.ndarray): Feasible matchings between drivers/passengers.
            travel_demand (numpy.ndarray): Travel demand.
            vehicle_supply (numpy.ndarray): Vehicle supply.
            road (RoadNetwork object): Road network object.

        Returns:
            np.ndarray: Shortest path pick times, reshaped (vehicle_supply size, travel_demand size).
        """
        pick_times = road.shortest_path_lengths(
            feasible_matching[:, 11], feasible_matching[:, 28], imp_name='times'
        )
        times_array = np.array(pick_times).reshape(vehicle_supply.shape[0], travel_demand.shape[0])
        return times_array

    def nearest_dispatch(self, travel_demand, vehicle_supply, distances_array, times_array):
        """
        Dispatch based on the nearest available vehicle for the passenger or vice versa.

        Parameters:
            travel_demand (np.ndarray): A numpy array representing the travel demand.
            vehicle_supply (np.ndarray): A numpy array representing the vehicle supply.
            distances_array (np.ndarray): A numpy array representing the shortest path distances between all pairs.
            times_array (np.ndarray): A numpy array representing the shortest path times between all pairs.

        Returns:
            tuple: Four lists containing the matched passenger indices, matched driver indices,
                    match distances and pick times.
        """
        if vehicle_supply.shape[0] < travel_demand.shape[0]:  # passengers > vehicles
            match_passenger_list, match_driver_list, match_distance_list, pick_times_list = self.nearest_match(
                range(vehicle_supply.shape[0]), distances_array, times_array)
        else:  # vehicles >= passengers
            match_driver_list, match_passenger_list, match_distance_list, pick_times_list = self.nearest_match(
                range(travel_demand.shape[0]), distances_array.T, times_array.T)
        return match_passenger_list, match_driver_list, match_distance_list, pick_times_list

    def nearest_match(self, target_indices, distance_array, time_array=None):
        """
        Match each target index to the closest available match.

        Parameters:
            target_indices (list, range or ndarray): Indices to be matched.
            distance_array (np.ndarray): A numpy array representing the shortest path distances.
            time_array (np.ndarray, optional): A numpy array representing the shortest path times.

        Returns:
            tuple: Four lists containing the matched indices, matched list,
                   match distances and pick times (if dynamic_impedance is True).
        """
        match_indices = []
        match_list = []
        match_distances = []
        pick_times = []
        for i in target_indices:
            distance = distance_array[i, :]
            distance[match_list] = 20000
            distance_mask = np.where((distance == distance.min()) &
                                     (distance <= self.max_match_radius))[0]

            if self.dynamic_impedance:
                pick_time = time_array[i, :]
                time_mask = np.union1d(np.array(match_list), distance_mask)
                pick_time[time_mask] = 100
                match_index = np.argwhere((pick_time == pick_time.min()) &
                                          (pick_time <= self.max_pick_times))
            else:
                match_index = np.argwhere((distance == distance.min()) &
                                          (distance <= self.max_match_radius))

            if match_index.shape[0] != 0:
                match_indices.append(i)
                match_list.append(int(match_index[0]))
                match_distances.append(float(distance[match_index][0]))
                if self.dynamic_impedance:
                    pick_times.append(float(pick_time[match_index][0]))

        if self.dynamic_impedance:
            return match_indices, match_list, match_distances, pick_times
        else:
            return match_indices, match_list, match_distances, match_distances

    def execute_dispatch(self, travel_demand, vehicle_supply, distances_array, times_array, algorithm):
        """
        Execute dispatch algorithm.

        Parameters:
            travel_demand (numpy.ndarray): A numpy array representing the travel demand.
            vehicle_supply (numpy.ndarray): A numpy array representing the vehicle supply.
            distances_array (numpy.ndarray): A numpy array representing the shortest path distances between all pairs.
            times_array (numpy.ndarray): A numpy array representing the shortest path times between all pairs.
            algorithm (str): Dispatch algorithm to be used. Options: 'nearest', 'bipartite_matching'.

        Returns:
            tuple: Four lists containing the matched passenger indices, matched driver indices,
                   match distances and pick times.

        Raises:
            NotImplementedError: If the input algorithm is not supported.
        """
        if algorithm == 'nearest':
            match_passenger_list, match_driver_list, match_distance_list, pick_times_list = self.nearest_dispatch(
                travel_demand, vehicle_supply, distances_array, times_array
            )
        elif algorithm == 'bipartite_matching':
            match_passenger_list, match_driver_list, match_distance_list, pick_times_list = self.bipartite_matching(
                travel_demand, vehicle_supply, distances_array, times_array
            )
        else:
            raise NotImplementedError
        return travel_demand[match_passenger_list], vehicle_supply[match_driver_list],\
            match_distance_list, pick_times_list

    def process_distance_array(self, distance_array):
        """
        Process the given distance array by replacing any distances greater than the max_match_radius with NaN.
        If the number of elements in the array is greater than or equal to nearest_num, apply nearest_mask.

        Parameters:
            distance_array (np.ndarray): The array containing the distances.

        Returns:
            distance_array (np.ndarray): The processed distance array.
        """
        distance_array[np.where(distance_array >= self.max_match_radius)] = np.nan
        if distance_array.shape[0] >= self.nearest_num:
            return nearest_mask(distance_array)
        return distance_array

    def process_mask_for_time_distance(self, distance_array, time_array):
        """
        Process distance and time arrays by replacing any values greater than the max_match_radius and max_pick_times
        with NaN. Apply nearest_mask if the number of elements in the array is greater than or equal to nearest_num.

        Parameters:
            distance_array (np.ndarray): The array containing the distances.
            time_array (np.ndarray): The array containing the times.

        Returns:
            tuple: The processed time and distance arrays.
        """
        distance_than_limitation = distance_array >= self.max_match_radius
        time_than_limitation = time_array >= self.max_pick_times
        all_limitation = distance_than_limitation | time_than_limitation
        time_array[all_limitation] = np.nan
        distance_array[all_limitation] = np.nan
        if time_array.shape[0] >= self.nearest_num:
            return nearest_mask_for_time_distance(time_array, distance_array)
        return time_array, distance_array

    def match_single_dynamic(self, matrix, driver_passenger_pair, another_matrix, match_object='passenger'):
        """
        Match single dynamic.

        Parameters:
            matrix (np.ndarray): The main matrix.
            driver_passenger_pair (tuple): The pair of driver and passenger indices.
            another_matrix (np.ndarray): Another matrix.
            match_object (str, optional): The object to match. Can be 'passenger' or 'driver'. Defaults to 'passenger'.

        Returns:
            tuple: The indices of the matched passenger and driver, and the minimum pick time and match distance.
        """
        if match_object == 'passenger':
            match_passenger_index = driver_passenger_pair[1]
            match_driver_index = np.array([driver_passenger_pair[0][np.argmin(matrix)]])
        else:
            match_passenger_index = np.array([driver_passenger_pair[1][np.argmin(matrix)]])
            match_driver_index = driver_passenger_pair[0]

        if self.dynamic_impedance:
            sum_match_distance = [np.min(another_matrix)]
            sum_pick_time = [np.min(matrix)]
        else:
            sum_match_distance = [np.min(matrix)]
            sum_pick_time = [sum_match_distance]
        return match_passenger_index, match_driver_index, sum_pick_time, sum_match_distance

    def bipartite_matching(self, travel_demand, vehicle_supply, distance_array, time_array=None):
        """
        Bipartite matching between passengers and drivers.

        Parameters:
            travel_demand (list or np.ndarray): The list of travel demand.
            vehicle_supply (list or np.ndarray): The list of vehicle supply.
            distance_array (np.ndarray): The array containing the distances.
            time_array (np.ndarray, optional): The array containing the times. Defaults to None.

        Returns:
            tuple: The lists of matched passenger indices, matched driver indices, matched distances, and pick times.
        """
        if self.dynamic_impedance:
            time_array, distance_array = self.process_mask_for_time_distance(distance_array, time_array)
            split_time_array = bfs_split_graph(time_array, self.max_pick_times)
            split_distance_array = bfs_split_graph(distance_array, self.max_match_radius)
            reconstruct_time_array = reconstruct_matrix(split_time_array, time_array)
            reconstruct_distance_array = reconstruct_matrix(split_time_array, distance_array)
        else:
            distance_array = self.process_distance_array(distance_array)
            split_distance_array = bfs_split_graph(distance_array, self.max_match_radius)
            reconstruct_distance_array = reconstruct_matrix(split_distance_array, distance_array)

        match_passenger_index, match_driver_index = np.array([]), np.array([])
        match_distance_list, pick_times_list = [], []
        for matrix_id, matrix in enumerate(reconstruct_time_array if self.dynamic_impedance
                                           else reconstruct_distance_array):
            if matrix.shape[1] == 1:
                mpi, mdi, spt, smd = self.match_single_dynamic(
                    matrix, split_time_array[matrix_id], reconstruct_distance_array[matrix_id], 'passenger'
                ) if self.dynamic_impedance else self.match_single_dynamic(
                    matrix, split_distance_array[matrix_id], None, 'passenger'
                )
            elif matrix.shape[0] == 1:
                mpi, mdi, spt, smd = self.match_single_dynamic(
                    matrix, split_time_array[matrix_id], reconstruct_distance_array[matrix_id], 'driver'
                ) if self.dynamic_impedance else self.match_single_dynamic(
                    matrix, split_distance_array[matrix_id], None, 'driver'
                )
            else:
                if matrix.shape[0] <= matrix.shape[1]:
                    match_drivers, match_passengers = solve_dense(matrix)
                else:
                    match_passengers, match_drivers = solve_dense(matrix.T)

                smd = reconstruct_distance_array[matrix_id][match_drivers, match_passengers].tolist()
                spt = matrix[match_drivers, match_passengers].tolist() if self.dynamic_impedance else smd
                mpi = split_distance_array[matrix_id][1][match_passengers]
                mdi = split_distance_array[matrix_id][0][match_drivers]

            match_distance_list += smd
            pick_times_list += spt
            match_passenger_index = np.concatenate((match_passenger_index, mpi))
            match_driver_index = np.concatenate((match_driver_index, mdi))

        match_passenger_index = match_passenger_index.astype(int)
        match_driver_index = match_driver_index.astype(int)
        return match_passenger_index.tolist(), match_driver_index.tolist(), match_distance_list, pick_times_list

    def dispatch(self, travel_demand, vehicle_supply, road):
        """
        Dispatch vehicles to meet travel demand.

        Parameters:
            travel_demand (list or np.ndarray): The list of travel demand.
            vehicle_supply (list or np.ndarray): The list of vehicle supply.
            road (RoadNetwork object): The road network object.

        Returns:
            tuple: The lists of matched passenger names, matched driver names, matched distances, and pick times.
        """
        if (len(travel_demand) == 0) or (len(vehicle_supply) == 0):
            return [], [], 0, 0

        feasible_matching = self.get_feasible_matching(travel_demand, vehicle_supply)

        # 在不同类中，计算距离或时间的部分需要重写
        distance_array = self.calculate_distance(feasible_matching, travel_demand, vehicle_supply, road)
        times_array = self.calculate_pick_times(feasible_matching, travel_demand, vehicle_supply, road)

        match_passenger_name, match_driver_name, match_distance_list, pick_time_list = self.execute_dispatch(
            travel_demand, vehicle_supply, distance_array, times_array, self.dispatch_algorithm
        )
        return match_passenger_name, match_driver_name, match_distance_list, pick_time_list

    def match_state_update(self, match_passenger_name, match_driver_name, match_distance_list, pick_time_list):
        """
        Update the state of the matched passengers and drivers.

        Parameters:
            match_passenger_name (list): The list of matched passenger names.
            match_driver_name (list): The list of matched driver names.
            match_distance_list (list): The list of matched distances.
            pick_time_list (list): The list of pick up times.
        """
        self.drivers.array[match_driver_name, 5] = np.array(match_passenger_name)  # 更新司机所匹配到的乘客的index
        self.drivers.array[match_driver_name, 6] = 1  # 司机工作状态为接客中
        # self.drivers.array[match_driver_name, 7] = 0  # 将司机的累计空闲时间重置为0
        self.drivers.array[match_driver_name, 8] += self.passengers.array[match_passenger_name, 8] * \
                                                    np.where(self.drivers.array[match_driver_name, 10] == 1,
                                                    1 - self.commission_ratio, 0)
        self.drivers.array[match_driver_name, 9] += 1  # 司机完成请求数+1
        self.drivers.array[match_driver_name, 12] = self.passengers.array[match_passenger_name, 14]  # 司机终点为乘客起点

        self.passengers.array[match_passenger_name, 10] = 4  # 乘客状态为接客中
        self.passengers.array[match_passenger_name, 11] = np.array(match_driver_name)  # 更新乘客所匹配到的司机的index
        self.passengers.array[match_passenger_name, 12] = match_distance_list
        self.passengers.array[match_passenger_name, 13] = pick_time_list

        self.step_match_distance = np.mean(match_distance_list) if match_distance_list else -1
        self.step_pick_times = np.mean(pick_time_list) if pick_time_list else -1

    def update_matched_driver_trajectories(self, passenger, driver, road):
        """
        Update the trajectories of the matched drivers.

        Parameters:
            passenger (np.ndarray or list): The matched passenger name.
            driver (np.ndarray or list): The matched driver name.
            road (RoadNetwork object): The road object.
        """
        if (len(passenger) != 0) and (len(driver) != 0):
            # 获取接客路径
            pickup_trajectories, pickup_route = road.get_trajectories(
                self.drivers.nodes(driver),
                self.passengers.dep_nodes(passenger),
                self.drivers.position(driver),
                self.passengers.dep_coords(passenger),
                self.dynamic_impedance
            )
            # 获取送客路径
            delivery_trajectories, delivery_route = road.get_trajectories(
                self.passengers.dep_nodes(passenger),
                self.passengers.arr_nodes(passenger),
                self.passengers.dep_coords(passenger),
                self.passengers.arr_coords(passenger),
                self.dynamic_impedance
            )

            # 存储接客路径与送客路径
            self.drivers.update_pick_trajectories_and_route(driver, pickup_trajectories, pickup_route)
            self.drivers.update_delivery_trajectories_and_route(driver, delivery_trajectories, delivery_route)

    def unmatch_state_update(self, idle_passengers, idle_vehicles, match_passenger_name, match_driver_name):
        """
        Update the state of the unmatched passengers and drivers.

        Parameters:
            idle_passengers (list): The list of idle passengers.
            idle_vehicles (list): The list of idle vehicles.
            match_passenger_name (list): The list of matched passenger names.
            match_driver_name (list): The list of matched driver names.
        """
        unmatch_vehicles_name = idle_vehicles[np.logical_not(np.isin(idle_vehicles, match_driver_name))]
        unmatch_passengers_name = idle_passengers[np.logical_not(np.isin(idle_passengers, match_passenger_name))]

        self.drivers.array[unmatch_vehicles_name, 7] += 1  # 没有被匹配的司机的空闲时间+1
        self.passengers.array[unmatch_passengers_name, 9] += 1  # 没有被匹配的乘客的等待时间+1

    def driver_state_check(self, env_time):
        """
        Check the state of the drivers and update as necessary.

        Parameters:
            env_time (float): The current time in the environment.
        """
        self.update_pickup_driver_position()
        self.update_delivery_drivers_position()
        self.update_offline_driver(env_time)

    def update_offline_driver(self, env_time):
        """
        Update the state of the drivers that have gone offline.

        Parameters:
            env_time (float): The current time in the environment.
        """
        offline = self.drivers.will_offline(env_time)
        self.drivers.array[offline, 6] = 4

    def update_pickup_driver_position(self):
        """
        Update the position of the drivers who are in the process of picking up passengers.
        """
        to_delivery = self.drivers.update_pickup_position()
        self.passengers.pick2serve(to_delivery)

    def update_delivery_drivers_position(self):
        """
        Update the position of the drivers who are in the process of delivering passengers.
        """
        to_finish = self.drivers.update_delivery_position()
        self.passengers.serve2finish(to_finish)

    def passenger_state_check(self):  # 更新乘客取消、最新的乘客前往上车点、更新这些乘客的位置。还需要激活哪些未被激活的乘客
        """
        Check the state of the passengers and update as necessary.
        """
        self.step_cancel_requests = self.passengers.update_canceled()
        self.update_passenger_onboard_position()

    def update_passenger_onboard_position(self):  # 更新上车乘客的状态
        """
        Update the position of the passengers who are onboard the vehicles.
        """
        self.passengers.update_onboard_position()

    def post_dispatch_statistics(self, travel_demand, vehicle_supply, match_passenger_name, match_driver_name):
        """
        Post dispatch statistics update based on the completed matches.

        Parameters:
            travel_demand (list): The list of passenger requests.
            vehicle_supply (list): The list of available drivers.
            match_passenger_name (list): The list of matched passenger names.
            match_driver_name (list): The list of matched driver names.
        """
        self.step_platform_revenue = np.sum(self.passengers.array[match_passenger_name, 8] *
                                            np.where(self.drivers.array[match_driver_name, 10] == 1,
                                                     self.commission_ratio, 1.0))
        self.step_completed_requests = len(match_passenger_name)
        self.step_order_response_rate = self.step_completed_requests / len(travel_demand) \
            if len(travel_demand) != 0 else 0
        self.step_cancel_rate = self.step_cancel_requests / len(travel_demand) if len(travel_demand) != 0 else 0

        self.step_waiting_times = np.mean(self.passengers.array[match_passenger_name, 9]) \
            if len(match_passenger_name) != 0 else 0

    def step_metrics_statistics(self):
        """
        Records statistics for the current time step.

        It appends the metrics such as platform revenue, completed requests, order response rate, cancel requests,
        cancel rate,
        match distance, pick times, and waiting times for the current step to their respective lists.
        """
        self.step_platform_revenue_list.append(self.step_platform_revenue)
        self.step_completed_requests_list.append(self.step_completed_requests)
        self.step_order_response_rate_list.append(self.step_platform_revenue)
        self.step_cancel_requests_list.append(self.step_cancel_requests)
        self.step_cancel_rate_list.append(self.step_cancel_rate)
        self.step_match_distance_list.append(self.step_match_distance)
        self.step_pick_times_list.append(self.step_pick_times)
        self.step_waiting_times_list.append(self.step_waiting_times)

    def final_statistics(self):
        """
        Perform final statistics update at the end of the simulation or episode.
        """
        finish_passenger = self.passengers.completed
        cancel_passenger = self.passengers.canceled

        self.platform_revenue += np.sum(self.passengers.array[finish_passenger, 8] *
                                 np.where(self.drivers.array[self.passengers.array[finish_passenger, 11].astype(int),
                                 9] == 1, self.commission_ratio, 1.0))
        self.completed_requests = len(finish_passenger)
        self.order_response_rate = self.completed_requests / self.passengers.num if self.passengers.num != 0 else 0
        self.cancel_requests = len(cancel_passenger)
        self.cancel_rate = self.cancel_requests / self.passengers.num if self.passengers.num != 0 else 0

        self.avg_match_distance = np.mean(self.passengers.array[:, 12]) if len(finish_passenger) != 0 else -1
        self.avg_pick_times = np.mean(self.passengers.array[:, 13]) if len(finish_passenger) != 0 else -1
        self.avg_waiting_times = np.mean(self.passengers.array[finish_passenger, 9]) if len(finish_passenger) != 0 else 0
        self.avg_driver_income = np.mean(self.drivers.array[:, 8])
        self.avg_driver_order = np.mean(self.drivers.array[:, 9])
        self.avg_driver_idle_duration = np.mean(self.drivers.array[:, 7])

    def get_match_object(self):
        """
        Retrieve the current passengers and drivers available for matching.

        Returns:
            tuple: A tuple containing two lists: passengers and drivers available for matching.
        """
        passengers = self.waiting_passengers()
        drivers = self.idle_drivers()
        return passengers, drivers

    def pre_dispatch_statistics(self):
        """
        Update statistics before the dispatch operation.
        """
        self.waiting_passenger_num = len(self.passengers.waiting_service)
        self.on_board_passenger_num = len(self.passengers.during_onboard)
        self.during_pick_passenger_num = len(self.passengers.during_pick)
        self.during_delivery_passenger_num = len(self.passengers.during_delivery)

        self.idle_driver_num = len(self.drivers.idle)
        self.pick_driver_num = len(self.drivers.pick)
        self.delivery_driver_num = len(self.drivers.delivery)
        self.reposition_driver_num = len(self.drivers.reposition)

    def activate_passengers_drivers(self, env_time, road):
        self.passengers.activate_passengers(env_time)
        self.drivers.begin_work(env_time)

    def pre_dispatch(self, env_time, road):
        """
        Preprocess the dispatch by activating passengers and drivers, updating their nodes and travel information.

        Parameters:
            env_time (int): The current time in the environment.
            road (RoadNetwork object): The road network where the dispatching happens.
        """
        self.activate_passengers_drivers(env_time, road)
        # 新出现的乘客前往上车点
        self.passenger_to_onboard(road)

        # 更新乘客预期出行时长、出行里程和出行费用等信息
        self.update_passenger_travel_info(road)
        self.pre_dispatch_statistics()

    def dispatch_post_dispatch(self, env_time, road):
        """
        Perform dispatching by matching passengers and drivers, update their states and trajectories,
        and handle unmatched passengers and drivers.

        Parameters:
            env_time (int): The current time in the environment.
            road (RoadNetwork object): The road network where the dispatching happens.
        """
        travel_demand, vehicle_supply = self.get_match_object()
        match_passenger_name, match_driver_name, match_distances_list, pick_times_list = self.dispatch(
            travel_demand, vehicle_supply, road
        )

        # 更新匹配成功的司乘状态及其行驶轨迹
        self.match_state_update(match_passenger_name, match_driver_name, match_distances_list, pick_times_list)
        self.update_matched_driver_trajectories(match_passenger_name, match_driver_name, road)

        self.unmatch_state_update(travel_demand, vehicle_supply, match_passenger_name, match_driver_name)

        self.driver_state_check(env_time)
        self.passenger_state_check()
        self.post_dispatch_statistics(
            travel_demand, vehicle_supply, match_passenger_name, match_driver_name
        )
        self.step_metrics_statistics()

    def step(self, env_time, road):
        """
        Execute one time step in the simulation environment. Updates the status of passengers, drivers,
        and performs dispatching of idle drivers to waiting passengers.

        Parameters:
            env_time: Current time in the environment.
            road: The road network in the simulation environment.
        """
        self.pre_dispatch(env_time, road)
        self.dispatch_post_dispatch(env_time, road)

    def update_passenger_node(self, travel_demand, road):
        """
        Updates the node of the passengers based on their departure and arrival coordinates.

        Parameters:
            travel_demand: A list of passenger demand that is to be serviced.
            road: The road network in the simulation environment.
        """
        self.passengers.array[travel_demand, 14] = road.get_coord_node(self.passengers.dep_coords(travel_demand))
        self.passengers.array[travel_demand, 15] = road.get_coord_node(self.passengers.arr_coords(travel_demand))

    def update_driver_node(self, vehicle_supply, road):
        """
        Updates the node of the drivers based on their current position.

        Parameters:
            vehicle_supply: A list of available drivers that can be dispatched to service passengers.
            road: The road network in the simulation environment.
        """
        self.drivers.array[vehicle_supply, 11] = road.get_coord_node(self.drivers.position(vehicle_supply))

    def passenger_to_onboard(self, road):
        """
        Updates the status of passengers who need to onboard to status 2 (onboard)
        and calculates the walking distance to the nearest node.

        Parameters:
            road: The road network in the simulation environment.
        """
        travel_demand = self.passengers.need_onboard
        if travel_demand.shape[0]:
            nearest_node = self.passengers.dep_nodes(travel_demand)
            dep_coords = self.passengers.dep_coords(travel_demand)
            walk_distance, walk_trajectories = road.calculate_walk_distance_coord(dep_coords, nearest_node)
            self.passengers.update_onboard_trajectories(travel_demand, walk_trajectories)
        self.passengers.array[travel_demand, 10] = 2

    def passenger_nodes(self):
        """
        Returns the departure nodes of all waiting passengers.

        Returns:
            np.ndarray: Array of departure nodes of the waiting passengers.
        """
        travel_demand = self.passengers.wait_pick
        return self.passengers.dep_nodes(travel_demand)

    def driver_routes(self):
        """
        Returns the routes of all operating drivers.

        Returns:
            np.ndarray: A list of routes for each operating driver on the road.
        """
        vehicle_supply = self.drivers.operating_on_road
        return self.drivers.routes(vehicle_supply)

    def step_trajectories(self):
        """
        Gets the trajectories of drivers and passengers for the current time step.

        For drivers, it extracts data from the 'drivers' object for operating drivers.
        For passengers, it extracts data from the 'passengers' object for those who are walking, waiting, or being
        picked up.

        Returns:
            drivers_trajectories, passengers_trajectories: Two arrays containing the trajectories for drivers and
            passengers, respectively.
        """
        drivers = self.drivers.operating
        drivers_trajectories = self.drivers.array[drivers, :][:, [1, 2, 6]]
        passengers = self.passengers.walk_wait_pick
        passengers_trajectories = self.passengers.array[passengers, :][:, [1, 2, 10]]
        return drivers_trajectories, passengers_trajectories

    def step_metrics(self):
        save_metrics = {
            'wait_passengers': self.waiting_passenger_num,
            'on_board_passengers': self.on_board_passenger_num,
            'pick_passengers': self.during_pick_passenger_num,
            'delivery_passenger': self.during_delivery_passenger_num,
            'idle_driver': self.idle_driver_num,
            'pick_driver': self.pick_driver_num,
            'delivery_driver': self.delivery_driver_num,
            'reposition_driver': self.reposition_driver_num,
            'platform_revenue': self.step_platform_revenue,
            'completed_requests': self.step_completed_requests,
            'order_response_rate': self.step_order_response_rate,
            'cancel_requests': self.step_cancel_requests,
            'cancel_rate': self.step_cancel_rate,
            'match_distance': self.step_match_distance,
            'pick_times': self.step_pick_times,
            'waiting_times': self.step_waiting_times
        }
        return save_metrics


class RoadChangedPlatform(Platform):
    def __init__(
            self, name: str, operate_type: str, passenger_path: str, driver_path: str, base_fare: float,
            unit_time_fare: float, unit_mileage_fare: float, commission_ratio: float, max_match_radius: float,
            max_pick_times: float, dispatch_algorithm: str, reposition_algorithm: str, nearest_limitation: int,
            simulation_type: str, dynamic_impedance: bool
    ):
        """
        This class extends the base class 'Platform' to model a ride-hailing platform where road changes occur
        (e.g., due to road closures or accidents). This could result in drivers changing their routes, and passengers
        needing to walk additional distances. It provides a number of metrics to track these changes.

        Parameters:
            name (str): Name of the platform.
            operate_type (str): Operating type of the platform.
            passenger_path (str): Path to the passenger data.
            driver_path (str): Path to the driver data.
            base_fare (float): Base fare for a ride.
            unit_time_fare (float): Fare per unit time for a ride.
            unit_mileage_fare (float): Fare per unit mileage for a ride.
            commission_ratio (float): Commission ratio for the platform.
            max_match_radius (float): Maximum matching radius.
            max_pick_times (float): Maximum pick-up time.
            dispatch_algorithm (str): Dispatching algorithm used by the platform.
            reposition_algorithm (str): Repositioning algorithm used by the platform.
            nearest_limitation (int): Limitation on the number of nearest nodes.
            simulation_type (str): Type of simulation.
            dynamic_impedance (bool): If True, the road impedance is dynamic.

        Attributes:
        control_driver_back_idle_num (int): Number of drivers that return to idle state.
        control_driver_change_route_num (int): Number of drivers that change their route.
        control_driver_no_path_cancel_num (int): Number of drivers that cancel their trip due to lack of path.
        control_driver_need_reposition_num (int): Number of drivers that need to be repositioned.
        control_driver_change_destination_num (int): Number of drivers that change their destination.
        control_passenger_additional_walk_num (int): Number of passengers that need to walk additional distance.
        control_activate_passenger_request_num (int): Number of passenger requests that get activated.
        control_unactivated_passenger_request_num (int): Number of passenger requests that remain unactivated.
        """
        super().__init__(name, operate_type, passenger_path, driver_path, base_fare,
                         unit_time_fare, unit_mileage_fare, commission_ratio, max_match_radius,
                         max_pick_times, dispatch_algorithm, reposition_algorithm, nearest_limitation,
                         simulation_type, dynamic_impedance)

        self.control_driver_back_idle_num = 0
        self.control_driver_change_route_num = 0
        self.control_driver_no_path_cancel_num = 0
        self.control_driver_need_reposition_num = 0
        self.control_driver_change_destination_num = 0
        self.control_passenger_additional_walk_num = 0
        self.control_activate_passenger_request_num = 0
        self.control_unactivated_passenger_request_num = 0

    def pre_dispatch(self, env_time, road):
        """
        Prepares the environment for dispatching by activating passengers and drivers,
        updating various state information, and handling changes to the road.

        Parameters:
            env_time (int): The current environment time.
            road (RoadNetwork object): The current road state.
        """
        self.activate_passengers_drivers(env_time, road)
        self.handle_road_changed(env_time, road)

        self.passenger_to_onboard(road)

        self.update_passenger_travel_info(road)
        self.pre_dispatch_statistics()

    def dispatch_post_dispatch(self, env_time, road):
        """
        Handles the post-dispatch operations, including matching state updates, driver and
        passenger state checks, and statistics gathering.

        Parameters:
            env_time (int): The current environment time.
            road (RoadNetwork object): The current road state.
        """
        travel_demand, vehicle_supply = self.get_match_object()
        # print('Vehicles: {}，Passengers: {}.'.format(vehicle_supply.shape[0], travel_demand.shape[0]))
        match_passenger_name, match_driver_name, match_distance_list, pick_times_list = self.dispatch(
            travel_demand, vehicle_supply, road
        )

        self.match_state_update(match_passenger_name, match_driver_name, match_distance_list, pick_times_list)
        self.update_matched_driver_trajectories(match_passenger_name, match_driver_name, road)

        self.unmatch_state_update(travel_demand, vehicle_supply, match_passenger_name, match_driver_name)

        self.driver_state_check(env_time)
        self.passenger_state_check()
        self.post_dispatch_statistics(
            travel_demand, vehicle_supply, match_passenger_name, match_driver_name
        )
        self.step_metrics_statistics()

    def step(self, env_time, road):
        """
        Performs a single step of the dispatching process by executing the pre-dispatch and post-dispatch methods.

        Parameters:
            env_time (int): The current environment time.
            road (RoadNetwork object): The current road state.
        """
        self.pre_dispatch(env_time, road)
        self.dispatch_post_dispatch(env_time, road)

    def handle_idle_vehicles_out_control_area(self, road):
        """
        Handles the idle vehicles that are out of the control area.
        The vehicles are repositioned and their state is updated accordingly.

        Parameters:
            road (RoadNetwork object): The current road state.
        """
        idle_vehicles = self.drivers.idle
        ivs_in_control = self.judge_driver_in_control_area(idle_vehicles, road)
        need_reposition_idle_vehicles = idle_vehicles[ivs_in_control]
        if need_reposition_idle_vehicles.size != 0:
            reposition_trajectories, reposition_routes = self.reposition_vehicle_out_control_area(
                need_reposition_idle_vehicles, road
            )
            self.drivers.update_reposition_trajectories_route(
                need_reposition_idle_vehicles, reposition_trajectories, reposition_routes
            )
            # 此时还需要更新车辆状态为调度中
            self.control_driver_need_reposition_num += len(need_reposition_idle_vehicles)
            self.drivers.array[need_reposition_idle_vehicles, 6] = 3

    def handle_pick_vehicles_destination_in_control(self, pick_vehicles, road, can_walk=False):
        """
        Handles the vehicles in pick-up state with a destination in the control area.
        The pick-up is cancelled and the state of vehicles and passengers is updated accordingly.

        Args:
            road (RoadNetwork object): The current road state.
            can_walk (bool): If set to True, the passengers can walk after control.
        """
        pvs_destination_in_control = self.judge_driver_destination_in_control_area(
            pick_vehicles, road, state='pick'
        )
        need_cancel_pick_vehicles = pick_vehicles[pvs_destination_in_control]
        need_cancel_passengers = self.drivers.matched_passenger(need_cancel_pick_vehicles)
        self.drivers.array[need_cancel_pick_vehicles, 5] = -1
        self.drivers.array[need_cancel_pick_vehicles, 6] = 0
        self.drivers.array[need_cancel_pick_vehicles, 7] += 1
        self.drivers.array[need_cancel_pick_vehicles, 12] = -1

        self.passengers.array[need_cancel_passengers, 10] = 7 if can_walk is False else 1
        self.passengers.array[need_cancel_passengers, 11] = -1

        self.control_driver_back_idle_num += len(need_cancel_pick_vehicles)
        return pick_vehicles[np.logical_not(pvs_destination_in_control)]

    def handle_vehicles__destination_not_control__route_control__have_path(
            self, vehicles, env_time, road, state
    ):
        """
        Handles vehicles in a given state ('pick' or 'delivery'), checks and updates their routes.

        Parameters:
            vehicles (np.ndarray or list): List of vehicles.
            env_time (int): Environmental time.
            road (RoadNetwork object): Road information.
            state (str): Operation state, 'pick' or 'delivery'.

        Returns:
            np.ndarray or list: Vehicles that need to be rerouted but do not have a valid path.

        Note:
             In this function, the destination of the vehicles is not in the control area, and the path may pass
             through the control area.
        """
        routes_though_control_area = self.judge_driver_route_in_control_area(
            vehicles, road, env_time, state
        )
        vehicles_right_now_in_control = self.judge_driver_in_control_area(vehicles, road)
        need_reformulate_route = routes_though_control_area & np.logical_not(vehicles_right_now_in_control)
        vehicle_re_route = vehicles[need_reformulate_route]
        driver_nodes = self.drivers.nodes(vehicle_re_route)
        driver_position = self.drivers.position(vehicle_re_route)

        if state == 'pick':
            destination_nodes = self.passengers.dep_nodes(self.drivers.matched_passenger(vehicle_re_route))
            destination_position = self.passengers.dep_coords(self.drivers.matched_passenger(vehicle_re_route))
            update_func = self.drivers.update_pick_trajectories_and_route
        elif state == 'delivery':
            destination_nodes = self.passengers.arr_nodes(self.drivers.matched_passenger(vehicle_re_route))
            destination_position = self.passengers.arr_coords(self.drivers.matched_passenger(vehicle_re_route))
            update_func = self.drivers.update_delivery_trajectories_and_route
        else:
            raise NotImplementedError("Wrong state input. Support 'pick' and 'delivery'.")

        vehicle_re_route_have_path = road.judge_od_have_path(
            driver_nodes, destination_nodes, self.dynamic_impedance
        )
        vehicles_have_path = vehicle_re_route[vehicle_re_route_have_path]
        new_trajectories, new_routes = road.get_trajectories(
            driver_nodes[vehicle_re_route_have_path], destination_nodes[vehicle_re_route_have_path],
            driver_position[vehicle_re_route_have_path], destination_position[vehicle_re_route_have_path],
            self.dynamic_impedance
        )

        update_func(vehicles_have_path, new_trajectories, new_routes)
        self.control_driver_change_route_num += len(vehicles_have_path)
        return vehicle_re_route[np.logical_not(vehicle_re_route_have_path)]

    def handle_vehicles__destination_not_control__route_control__no_path(self, vehicles):
        """
        Handles the vehicles with a destination not in control, route in control, but no path.
        The vehicle and passenger states are updated accordingly.

        Parameters:
            vehicles (np.ndarray): The vehicles to be handled.
        """
        no_path_passengers = self.drivers.matched_passenger(vehicles)
        self.drivers.array[vehicles, 5] = -1
        self.drivers.array[vehicles, 6] = 0
        self.drivers.array[vehicles, 7] += 1
        self.drivers.array[vehicles, 12] = -1

        self.passengers.array[no_path_passengers, 10] = 7
        self.passengers.array[no_path_passengers, 11] = -1

        self.control_driver_no_path_cancel_num += len(vehicles)
        self.control_driver_back_idle_num += len(vehicles)

    def handle_delivery_vehicles_destination_in_control(self, delivery_vehicles, road):
        """
        Handles the vehicles currently in delivery state and whose destination lies in the control area.
        The vehicles are repositioned and their trajectories and routes are updated accordingly.

        Parameters:
            delivery_vehicles (np.ndarray): Drivers who are in the process of loading passengers
            road (RoadNetwork object): Road information.
        Returns:
            np.ndarray or list : The vehicles whose destination is not in the control area.
        """
        dvs_destination_in_control = self.judge_driver_destination_in_control_area(
            delivery_vehicles, road, state='delivery'
        )
        need_reposition_delivery_vehicles = delivery_vehicles[dvs_destination_in_control]
        if need_reposition_delivery_vehicles.size != 0:
            reposition_trajectories, reposition_routes = self.reposition_vehicle_out_control_area(
                need_reposition_delivery_vehicles, road
            )
            self.drivers.update_delivery_trajectories_and_route(
                need_reposition_delivery_vehicles, reposition_trajectories, reposition_routes
            )
        self.control_driver_change_destination_num += len(need_reposition_delivery_vehicles)
        return delivery_vehicles[np.logical_not(dvs_destination_in_control)]

    def handle_road_changed(self, env_time, road):
        """
        Responds to changes in road conditions and modifies the behavior of vehicles in response to these changes.
        Depending on whether vehicles and passengers are allowed to drive or walk, it makes decisions to reposition vehicles,
        cancel passenger requests, or allow passengers to walk to their destinations.

        Args:
            env_time (int): The current environment time.
            road (RoadNetwork object): Road information.
        """
        can_walk, can_drive = road.current_can_walk, road.current_can_drive
        if (can_drive is False) and (can_walk is False):
            # 如果区域内车辆不准行驶，乘客不许出行
            self.handle_idle_vehicles_out_control_area(road)

            # 接客车辆。
            # 1. 如果它会经过这个区域，但还不在该区域内，且它的终点不停留在区域内，那么需要重新规划路线。--对应乘客不受影响。
            # 2. 如果他经过这个区域，且已在该区域内，且它的终点不停留在区域内，那么正常行驶。--对应乘客不受影响。
            # 3. 如果他终点在区域内，需要取消订单。--对应乘客取消出行。
            # 终点在管控区
            pick_vehicles = self.drivers.pick
            if pick_vehicles.size != 0:
                pvs_destination_not_in_control = self.handle_pick_vehicles_destination_in_control(pick_vehicles, road)
                no_path_vehicles = self.handle_vehicles__destination_not_control__route_control__have_path(
                    pvs_destination_not_in_control, env_time, road, 'pick'
                )
                self.handle_vehicles__destination_not_control__route_control__no_path(no_path_vehicles)

            # 载客车辆
            # 1. 如果它只是经过这个区域，但还不在该区域内，终点不停留在区域内，那么需要重新规划路线--对应乘客不受影响。
            # 2. 如果他经过这个区域，且已在该区域内，且它的终点不停留在区域内，那么正常行驶。--对应乘客不受影响。
            # 3. 如果他终点在区域内，则需要行驶到距离终点最近的非管控点。--对应乘客只能抵达新的目的地。
            delivery_vehicles = self.drivers.delivery
            if delivery_vehicles.size != 0:
                dvs_destination_not_in_control = self.handle_delivery_vehicles_destination_in_control(
                    delivery_vehicles, road)
                no_path_vehicles = self.handle_vehicles__destination_not_control__route_control__have_path(
                    dvs_destination_not_in_control, env_time, road, 'delivery'
                )
                self.handle_vehicles__destination_not_control__route_control__no_path(no_path_vehicles)

            # 区域内的没上车（等待匹配的）的出行需求和潜在的出行需求全部取消。
            self.cancel_activate_passenger(road)
            self.cancel_unactivated_passenger(road)
        elif (can_drive is True) and (can_walk is False):  # 如果车辆允许行驶，但乘客不许出行
            # 空闲车辆
            # 1. 如果在区域内，需要驶往区域外。 2.如果不在，不受影响
            self.handle_idle_vehicles_out_control_area(road)

            # 接客车辆。
            # 1. 如果它会经过这个区域，但还不在该区域内，且它的终点不停留在区域内，那么正常行驶--对应乘客不受影响。
            # 2. 如果他经过这个区域，且已在该区域内，且它的终点不停留在区域内，那么正常行驶。--对应乘客不受影响。
            # 3. 如果他终点在区域内，需要取消订单。--对应乘客的出行需求被取消。
            pick_vehicles = self.drivers.pick
            if pick_vehicles.size != 0:
                self.handle_pick_vehicles_destination_in_control(pick_vehicles, road)

            # 载客车辆。
            # 1. 如果它只是经过这个区域，但还不在该区域内，终点不停留在区域内，那么正常行驶。--对应乘客不受影响。
            # 2. 如果他经过这个区域，且已在该区域内，且它的终点不停留在区域内，那么正常行驶。--对应乘客不受影响。
            # 3. 如果他终点在区域内，车辆正常行驶。--对应乘客不受影响。

            # 区域内的没上车（等待匹配的）的出行需求和潜在的出行需求全部取消。
            self.cancel_activate_passenger(road)
            self.cancel_unactivated_passenger(road)
        elif (can_drive is False) and (can_walk is True):  # 如果车辆不允许行驶，但乘客允许出行
            # 空闲车辆
            # 1. 如果在区域内，需要驶往区域外。 2.如果不在，不受影响
            self.handle_idle_vehicles_out_control_area(road)

            # 接客车辆
            # 1. 如果它会经过这个区域，但还不在该区域内，且它的终点不停留在区域内，那么需要重新规划路线。--对应乘客不受影响。
            # 2. 如果他经过这个区域，且已在该区域内，且它的终点不停留在区域内，那么正常行驶。--对应乘客不受影响。
            # 3. 如果他终点在区域内，需要取消订单。--对应乘客出行需求变为需要步行前往上车点的需求。
            # 终点在管控区
            pick_vehicles = self.drivers.pick
            if pick_vehicles.size != 0:
                pvs_destination_not_in_control = self.handle_pick_vehicles_destination_in_control(
                    pick_vehicles, road, can_walk=True)
                no_path_vehicles = self.handle_vehicles__destination_not_control__route_control__have_path(
                    pvs_destination_not_in_control, env_time, road, 'pick'
                )
                self.handle_vehicles__destination_not_control__route_control__no_path(no_path_vehicles)

            # 载客车辆。
            # 1. 如果它只是经过这个区域，但还不在该区域内，终点不停留在区域内，那么需要重新规划路线--对应乘客不受影响。
            # 2. 如果他经过这个区域，且已在该区域内，且它的终点不停留在区域内，那么正常行驶。--对应乘客不受影响。
            # 3. 如果他终点在区域内，则需要行驶到距离终点最近的非管控点。--对应乘客在到达终点后需要步行前往它的终点
            delivery_vehicles = self.drivers.delivery
            if delivery_vehicles.size != 0:
                dvs_destination_not_in_control = self.handle_delivery_vehicles_destination_in_control(
                    delivery_vehicles, road)
                no_path_vehicles = self.handle_vehicles__destination_not_control__route_control__have_path(
                    dvs_destination_not_in_control, env_time, road, 'delivery'
                )
                self.handle_vehicles__destination_not_control__route_control__no_path(no_path_vehicles)

            # 区域内没上车（等待匹配的、等待接客的）的出行需求将步行前往距离最近的非管控点后，再次提交出行请求。
            # 因为道路内不允许车辆行驶，还需要判断乘客的出行是否存在路径。
            self.activate_passenger_rewalk_to_nodes(road)
            self.cancel_activate_passenger(road)
            self.cancel_unactivated_passenger(road)
        else:  # 正常情况
            pass

    def judge_driver_in_control_area(self, vehicle_supply, road):
        """
        Determines if the given vehicle is in the control area.

        Parameters:
            vehicle_supply (np.ndarray or list) : The vehicles to be checked.
            road (RoadNetwork object): Road information.
        Returns:
            np.ndarray or list : An array indicating whether each vehicle is in the control area.
        """
        vehicle_nodes = self.drivers.nodes(vehicle_supply)
        driver_in_control_area = road.judge_nodes_in_control_area(vehicle_nodes)
        return driver_in_control_area

    def judge_driver_destination_in_control_area(self, vehicle_supply, road, state='pick'):
        """
        Evaluates if the destination of the provided vehicles is in the control area.

        Parameters:
            vehicle_supply (np.ndarray or list): The vehicles to be checked.
            road (RoadNetwork object): The current road state.
            state (str): The state of the vehicle, either 'pick' for pickup or 'delivery'.
        Returns:
            np.ndarray or list : An array indicating whether the destination of each vehicle is in the control area.
        """
        match_passengers = self.drivers.matched_passenger(vehicle_supply)
        destination = self.passengers.dep_nodes(match_passengers) if state == 'pick'\
            else self.passengers.arr_nodes(match_passengers)
        node_in_control = road.judge_nodes_in_control_area(destination)
        return node_in_control

    def judge_driver_route_in_control_area(self, vehicle_supply, road, env_time, state='pick'):
        """
        Determines if the route a vehicle is taking is passing through the control area.

        Parameters:
            vehicle_supply (np.ndarray or list): The vehicles to be checked.
            road (RoadNetwork object): The current road state.
            env_time (int): The current environment time.
            state (str): The state of the vehicle, either 'pick' for pickup or 'delivery'.
        Returns:
            np.ndarray or list : An array indicating whether the route of each vehicle is passing through the control area.
        """
        if state == 'pick':
            routes = [np.array(self.drivers.pickup_route_deque[vehicle]) for vehicle in vehicle_supply]
        else:
            routes = [np.array(self.drivers.delivery_route_deque[vehicle]) for vehicle in vehicle_supply]
        routes_though_control_area = road.judge_routes_in_control_area(routes, env_time)
        return routes_though_control_area

    def reposition_vehicle_out_control_area(self, vehicle_supply, road):
        """
        Calculates new trajectories and routes for vehicles that need to be moved out of the control area.

        Parameters:
            vehicle_supply (np.ndarray or list): The vehicles to be checked.
            road (RoadNetwork object): The current road state.
        Returns:
            The new trajectories and routes for the vehicles.
        """
        origin_nodes = self.drivers.nodes(vehicle_supply)
        origin_coordinates = self.drivers.position(vehicle_supply)
        # 这里是否需要判断下这个目前节点是否与车辆的当前节点是连通的？
        target_nodes = road.current_road.get_node_ids(origin_coordinates[:, 0], origin_coordinates[:, 1])
        target_coordinates = road.get_node_coordinates(target_nodes)
        reposition_trajectories, reposition_routes = road.get_trajectories_from_origin_road(
            origin_nodes, target_nodes, origin_coordinates, target_coordinates, self.dynamic_impedance
        )  # 这时候应该用之间的网络计算路线，因为现有的网络已经没有从 origin_nodes到target_nodes的路径了
        return reposition_trajectories, reposition_routes

    def activate_passenger_rewalk_to_nodes(self, road):
        """
        Recalculates new walking paths for passengers in the control area who need to reach a new node due to the
        vehicle restrictions in the control area.

        Parameters:
            road (RoadNetwork object): The current road state.
        """
        # 等待前往上车点、上车点途中，等待匹配
        travel_demand = np.where(np.isin(self.passengers.array[:, 10], [1, 2, 3]))[0]
        nodes = self.passengers.dep_nodes(travel_demand)

        node_in_control_area = road.judge_nodes_in_control_area(nodes)
        passenger_in_control_area = travel_demand[node_in_control_area]
        if len(passenger_in_control_area) != 0:
            passenger_coordinates = self.passengers.dep_coords(passenger_in_control_area)
            new_nodes = road.get_coord_node(passenger_coordinates)

            walk_distance, walk_trajectories = road.calculate_walk_distance_coord(passenger_coordinates, new_nodes)
            # 更新乘客的步行轨迹
            self.passengers.array[passenger_in_control_area, 14] = new_nodes
            self.passengers.update_onboard_trajectories(
                passenger_in_control_area, walk_trajectories
            )
            self.control_passenger_additional_walk_num += len(passenger_in_control_area)

    def cancel_activate_passenger(self, road):
        """
        Cancels the transportation request of passengers waiting to be picked up if they are affected by
        road control measures.

        Parameters:
            road (RoadNetwork object): The current road state.
        """
        # 这个需要判断当前时刻是否在管控时间内
        wait_walk_pick_passenger = self.passengers.walk_wait_pick  # 没判断这些乘客是否在区域内
        if len(wait_walk_pick_passenger) != 0:
            dep_nodes = self.passengers.dep_nodes(wait_walk_pick_passenger)
            arr_nodes = self.passengers.arr_nodes(wait_walk_pick_passenger)
            dep_node_in_control = road.judge_nodes_in_control_area(dep_nodes)  # 判断这些乘客是否受节点的影响
            arr_node_in_control = road.judge_nodes_in_control_area(arr_nodes)

            node_in_control = np.array(dep_node_in_control) | np.array(arr_node_in_control)
            self.passengers.control_cancel(wait_walk_pick_passenger[node_in_control])  # 这些乘客因此取消订单
            self.control_activate_passenger_request_num += len(wait_walk_pick_passenger[node_in_control])

            passenger_node_not_control = wait_walk_pick_passenger[np.logical_not(node_in_control)]

            # 因道路管控而缺少路径
            no_path_need_changed = road.judge_od_have_path(
                self.passengers.dep_nodes(passenger_node_not_control),
                self.passengers.arr_nodes(passenger_node_not_control),
                self.dynamic_impedance
            )
            # 这些乘客因此取消订单
            self.passengers.control_cancel(passenger_node_not_control[np.logical_not(no_path_need_changed)])
            self.control_activate_passenger_request_num += \
                len(passenger_node_not_control[np.logical_not(no_path_need_changed)])

    def cancel_unactivated_passenger(self, road):
        """
        Cancels the transportation request of passengers who haven't been activated yet,
        if their transportation plans are affected by road control measures.

        Parameters:
            road (RoadNetwork object): The current road state.
        """
        unactivated_passenger = self.passengers.unactivated
        dep_nodes = self.passengers.dep_nodes(unactivated_passenger)
        arr_nodes = self.passengers.arr_nodes(unactivated_passenger)

        dep_node_in_control = road.judge_nodes_in_control_area(dep_nodes)  # 判断这些乘客是否受节点的影响
        arr_node_in_control = road.judge_nodes_in_control_area(arr_nodes)
        node_in_control = np.array(dep_node_in_control) | np.array(arr_node_in_control)
        time_in_control = np.in1d(np.array(self.passengers.array[unactivated_passenger, 5]), road.now_change_time_range)

        passenger_need_changed = node_in_control & time_in_control
        self.passengers.control_cancel(unactivated_passenger[passenger_need_changed])
        self.control_unactivated_passenger_request_num += len(unactivated_passenger[passenger_need_changed])

    def driver_state_check(self, env_time):
        """
        Check the state of the drivers and update as necessary.

        Parameters:
            env_time (float): The current time in the environment.
        """
        # 过滤离线的司机并更改其工作状态
        self.update_offline_driver(env_time)

        self.update_pickup_driver_position()
        self.update_delivery_drivers_position()
        self.update_reposition_drivers_position()

    def update_reposition_drivers_position(self):
        """
        Update the position of the drivers who are in the process of reposition.
        """
        self.drivers.update_reposition_position()

