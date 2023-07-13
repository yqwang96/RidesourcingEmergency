# @Time : 2023/5/14 12:06
# @Author : Yinquan Wang<19114012@bjtu.edu.cn>
# @File : Environment.py
# @Function:
import os
import yaml
import numpy as np
from .RoadNetwork import RoadNetwork, RoadChangedRoadNetwork
from .Platform import Platform, RoadChangedPlatform


# 外部调用接口，
# 1. 指定派单算法类型
# 2. 观测状态编制（车辆位置、乘客位置） （网格的车辆数量、乘客数量，全局的车辆数量、乘客数量）
# 3. 动作输入
# 4. 奖励函数构建（完成出行需求数、平台收入、司机平均收入、匹配距离）

class Environment:
    def __init__(
            self, city: str, platform_path: str, drive_speed: float, walk_speed: float, time_limit: int,
            save_path: str
    ):
        """
        Initializes a new instance of the simulation environment.

        Parameters:
            city (str): The name of the city for simulation.
            platform_path (str): Path to the YAML file storing platform operating settings such as rate of commission,
                                 matching distance limit, billing standards, dispatch algorithm settings, etc.
            drive_speed (float): The driving speed in the city for simulation.
            walk_speed (float): The walking speed in the city for simulation.
            time_limit (int): The time limit for the simulation.

        Note:
            This function also initializes various indicators for monitoring and evaluating the simulation results.
        """
        self.city = city
        self.platform_path = platform_path  # 存储平台的运营设置。如抽成比例、匹配距离限制、接客时长限制、计费标准、派单算法设置

        self.drive_speed = drive_speed
        self.walk_speed = walk_speed
        self.time_limit = time_limit

        self.save_path = save_path
        self.road_network = RoadNetwork(self.city, self.drive_speed, self.walk_speed)
        self.platforms = self.load_platform()
        self.pre_update_nodes()

        self.time = 1

        # 评估指标(仿真完成后评估)
        self.market_revenue = 0            # 市场收入
        self.completed_requests = 0        # 完成请求数
        self.order_response_rate = 0       # 订单响应率
        self.cancel_requests = 0           # 取消请求数
        self.cancel_rate = 0               # 订单取消率
        self.avg_match_distance = 0        # 平均匹配距离
        self.avg_pick_times = 0            # 平均接客时长
        self.avg_waiting_times = 0         # 平均等待时长
        self.avg_driver_income = 0         # 平均司机收入
        self.avg_driver_order = 0          # 平均司机订单数

        # 监测指标(仿真进行中监测)
        self.waiting_passenger_num = 0             # 等待乘客数
        self.on_board_passenger_num = 0            # 前往上车点乘客数
        self.during_pick_passenger_num = 0         # 正在等待接客乘客数
        self.during_delivery_passenger_num = 0     # 正在前往目的地的乘客数

        self.idle_driver_num = 0                   # 空闲司机数
        self.pick_driver_num = 0                   # 接客司机数
        self.delivery_driver_num = 0               # 载客司机数
        self.reposition_driver_num = 0             # 调度司机数

        self.step_market_revenue = 0               # 单步市场收入
        self.step_completed_requests = 0           # 单步完成请求数
        self.step_order_response_rate = 0          # 单步订单响应率
        self.step_cancel_requests = 0              # 单步取消请求数
        self.step_cancel_rate = 0                  # 单步订单取消率
        self.step_match_distance = 0               # 单步平均匹配距离
        self.step_pick_times = 0                   # 单步平均接客时长
        self.step_waiting_times = 0                # 单步平均等待时长

        self.step_market_revenue_list = []
        self.step_completed_requests_list = []
        self.step_order_response_rate_list = []
        self.step_cancel_requests_list = []
        self.step_cancel_rate_list = []
        self.step_match_distance_list = []
        self.step_pick_times_list = []
        self.step_waiting_times_list = []

        self.create_platform_save_dir(self.save_path)

    def pre_update_nodes(self):
        [platform.drivers.update_nodes(self.road_network) for platform in self.platforms]
        [platform.passengers.update_dep_arr_nodes(self.road_network) for platform in self.platforms]

    @property
    def board(self):  # 查询当前仿真城市的经纬度边界
        """
        Returns the geographical boundaries of the city in the simulation.

        Returns:
            tuple: The geographical boundaries of the city, in (min_x, min_y, max_x, max_y) format.
        """
        return self.road_network.board

    def load_platform(self):
        """
        Loads the platform operating settings from the YAML file specified in platform_path.

        Returns:
            list: A list of Platform objects, each representing the settings of a platform.

        Note:
            The Platform class needs to be defined elsewhere and should take a dictionary of settings as parameters.
        """
        with open(self.platform_path, 'r') as f:
            platform_info = yaml.safe_load(f)
        return [Platform(**p) for p in platform_info]

    @property
    def road_nodes(self):
        """
        Returns the nodes of the road network in the city for simulation.

        Returns:
            list: A list of node indices in the road network.
        """
        return self.road_network.nodes

    @property
    def road_route(self):
        """
        Returns the routes of the road network in the city for simulation.

        Returns:
            list: A list of route indices in the road network.
        """
        return self.road_network.routes

    def node_passenger_counts(self):
        """
        Returns the count of waiting passengers at each node in the road network.

        Returns:
            list: A list of passenger counts corresponding to each node in the road network.
        """
        waiting_passenger_node = np.hstack([platform.passenger_nodes() for platform in self.platforms])
        counts = [np.count_nonzero(waiting_passenger_node == node) for node in self.road_nodes]
        return counts

    def route_driver_counts(self):
        """
        Returns the count of operating drivers on each route in the road network.

        Returns:
            list: A list of driver counts corresponding to each route in the road network.
        """
        operating_driver_route = np.hstack([platform.driver_routes() for platform in self.platforms])
        counts = [np.count_nonzero(operating_driver_route == route) for route in self.road_route]
        return counts

    def step(self):
        """
        Advances the simulation by one time step.

        This function performs the dispatching process for each platform, updates the statistics, resets the step-level
        statistics, and updates the time and road impedance for the next time step.
        """
        [platform.pre_dispatch(self.time, self.road_network) for platform in self.platforms]
        self.pre_statistics()

        [platform.dispatch_post_dispatch(self.time, self.road_network) for platform in self.platforms]
        self.step_save_trajectories()

        self.post_statistics()
        self.step_reset()
        self.update_to_next_time()

    def pre_statistics(self):
        """
        Updates the simulation-wide statistics before dispatching drivers and riders.

        This function aggregates statistics across all platforms.
        """
        for platform in self.platforms:
            self.waiting_passenger_num += platform.waiting_passenger_num
            self.on_board_passenger_num += platform.on_board_passenger_num
            self.during_pick_passenger_num += platform.during_pick_passenger_num
            self.during_delivery_passenger_num += platform.during_delivery_passenger_num

            self.idle_driver_num += platform.idle_driver_num
            self.pick_driver_num += platform.pick_driver_num
            self.delivery_driver_num += platform.delivery_driver_num
            self.reposition_driver_num += platform.reposition_driver_num

    def post_statistics(self):
        """
        Updates the simulation-wide statistics after dispatching drivers and riders.

        This function aggregates statistics across all platforms.
        """
        for platform in self.platforms:
            self.step_market_revenue += platform.step_platform_revenue
            self.step_completed_requests += platform.step_completed_requests
            self.step_order_response_rate += platform.step_order_response_rate
            self.step_cancel_requests += platform.step_cancel_requests
            self.step_cancel_rate += platform.step_cancel_rate
            self.step_match_distance += platform.step_match_distance
            self.step_pick_times += platform.step_pick_times
            self.step_waiting_times += platform.step_waiting_times

        self.step_order_response_rate = self.step_order_response_rate / len(self.platforms)
        self.step_cancel_rate = self.step_cancel_rate / len(self.platforms)
        self.step_match_distance = self.step_match_distance / len(self.platforms)
        self.step_pick_times = self.step_pick_times / len(self.platforms)
        self.step_waiting_times = self.step_waiting_times / len(self.platforms)

        self.step_market_revenue_list.append(self.step_market_revenue)
        self.step_completed_requests_list.append(self.step_completed_requests)
        self.step_order_response_rate_list.append(self.step_order_response_rate)
        self.step_cancel_requests_list.append(self.step_cancel_requests)
        self.step_cancel_rate_list.append(self.step_cancel_rate)
        self.step_match_distance_list.append(self.step_match_distance)
        self.step_pick_times_list.append(self.step_pick_times)
        self.step_waiting_times_list.append(self.step_waiting_times)

    def step_metrics(self):
        save_dict = {
            'wait_passengers': self.waiting_passenger_num,
            'onboard_passengers': self.on_board_passenger_num,
            'pick_passengers': self.during_pick_passenger_num,
            'delivery_passenger': self.during_delivery_passenger_num,

            'idle_driver': self.idle_driver_num,
            'pick_driver': self.pick_driver_num,
            'delivery_driver': self.delivery_driver_num,
            'reposition_driver': self.reposition_driver_num,

            'completed_requests': self.step_completed_requests,
            'canceled_requests': self.cancel_requests,
            'market_revenue (Yuan)': self.step_market_revenue,
            'order_response_rate (%)': self.step_order_response_rate,
            'cancel_rate (%)': self.step_cancel_rate,
            'match_distance (meter)': self.step_match_distance,
            'pick_times (min)': self.step_pick_times,
            'waiting_times (min)': self.step_waiting_times
        }
        return save_dict

    def final_statistics(self):
        """
        Calculates and updates the final statistics of the simulation when the simulation ends.

        This function aggregates the final statistics across all platforms.
        """
        [platform.final_statistics() for platform in self.platforms]
        for platform in self.platforms:
            self.market_revenue += platform.platform_revenue
            self.completed_requests += platform.completed_requests
            self.order_response_rate += platform.order_response_rate
            self.cancel_requests += platform.cancel_requests
            self.cancel_rate += platform.cancel_rate
            self.avg_match_distance += platform.avg_match_distance
            self.avg_pick_times += platform.avg_pick_times
            self.avg_waiting_times += platform.avg_waiting_times
            self.avg_driver_income += platform.avg_driver_income
            self.avg_driver_order += platform.avg_driver_order
        self.avg_match_distance = self.avg_match_distance / len(self.platforms)
        self.avg_pick_times = self.avg_pick_times / len(self.platforms)
        self.avg_waiting_times = self.avg_waiting_times / len(self.platforms)
        self.avg_driver_income = self.avg_driver_income / len(self.platforms)
        self.avg_driver_order = self.avg_driver_order / len(self.platforms)

    def update_road_impedance(self):  # 更新道路阻抗
        """
        Updates the impedance of each route in the road network based on the number of drivers on each route.

        This function is typically called at each time step to reflect the change in road conditions.
        """
        route_driver_num = self.route_driver_counts()
        self.road_network.update_impedance(route_driver_num)

    def update_to_next_time(self):
        """
        Updates the simulation to the next time step.

        This function first updates the road impedance based on the number of drivers, and then increments
        the time counter.
        """
        self.update_road_impedance()
        self.time += 1

    def create_platform_save_dir(self, path):
        """
        Creates necessary directories for saving the platform data.

        This function creates directories for the environment, metrics, platforms, drivers and passengers.

        Parameters:
            path (str) : The root directory where the platform directories will be created.
        """
        env_dir = os.path.join(path, r'{}'.format(str(self.__class__.__name__)))
        if not os.path.exists(env_dir):
            os.makedirs(env_dir)  # 为各个环境创建路径
            os.makedirs(os.path.join(env_dir, 'all_step_metrics'))
            os.makedirs(os.path.join(env_dir, 'single_step_metrics'))
        for platform in self.platforms:
            platform_dir = os.path.join(env_dir, r'{}'.format(platform.name))
            if not os.path.exists(platform_dir):
                os.makedirs(platform_dir)  # 为各个环境下的各个平台创建路径
                os.makedirs(os.path.join(platform_dir, r'drivers'))  # 各时间的平台车辆轨迹存储路径
                os.makedirs(os.path.join(platform_dir, r'passengers'))  # 各时间的平台乘客轨迹存储路径
                os.makedirs(os.path.join(platform_dir, r'step_metrics'))  # 各时间的平台统计指标存储路径

    def step_save_trajectories(self):
        """
        Saves the trajectories of drivers and passengers for the current time step.

        The function gets the trajectories from the platform's step_trajectories function and saves them in the platform
        directory.
        """
        env_dir = os.path.join(self.save_path, r'{}'.format(str(self.__class__.__name__)))
        for platform in self.platforms:
            platform_dir = os.path.join(env_dir, r'{}'.format(platform.name))
            drivers_trajectories, passenger_trajectories = platform.step_trajectories()
            platform_metrics = platform.step_metrics()

            metrics_save_path = os.path.join(platform_dir,
                                             r'step_metrics\metrics_time_{}.npy'.format(str(self.time - 1)))
            driver_save_path = os.path.join(platform_dir,
                                            r'drivers\drivers_time_{}.npy'.format(str(self.time - 1)))
            passenger_save_path = os.path.join(platform_dir,
                                               r'passengers\passengers_time_{}.npy'.format(str(self.time - 1)))

            np.save(metrics_save_path, platform_metrics)
            np.save(driver_save_path, drivers_trajectories)
            np.save(passenger_save_path, passenger_trajectories)
        env_step_metrics_save_path =\
            os.path.join(env_dir, r'single_step_metrics\metrics_time_{}.npy'.format(str(self.time - 1)))
        env_step_metrics = self.step_metrics()
        np.save(env_step_metrics_save_path, env_step_metrics)

    def step_reset(self):
        """
        Resets the step-level statistics to 0.

        This function is typically called at the end of each time step to prepare for the next time step.
        """

        self.waiting_passenger_num = 0             # 等待乘客数
        self.on_board_passenger_num = 0            # 前往上车点乘客数
        self.during_pick_passenger_num = 0         # 正在等待接客乘客数
        self.during_delivery_passenger_num = 0     # 正在前往目的地的乘客数

        self.idle_driver_num = 0                   # 空闲司机数
        self.pick_driver_num = 0                   # 接客司机数
        self.delivery_driver_num = 0               # 载客司机数
        self.reposition_driver_num = 0             # 调度司机数

        self.step_market_revenue = 0               # 单步市场收入
        self.step_completed_requests = 0           # 单步完成请求数
        self.step_order_response_rate = 0          # 单步订单响应率
        self.step_cancel_requests = 0              # 单步取消请求数
        self.step_cancel_rate = 0                  # 单步订单取消率
        self.step_match_distance = 0               # 单步平均匹配距离
        self.step_pick_times = 0                   # 单步平均接客时长
        self.step_waiting_times = 0                # 单步平均等待时长

    def save_platform_step_metrics(self, path):
        """
        Saves the step metrics for each platform and for the entire environment into .npz files. Each .npz file contains
        several arrays, each representing a different metric.

        Parameters:
            path (str): The directory where the .npz files will be saved. If the directory does not exist, it will be
            created.
        """
        for i, platform in enumerate(self.platforms):
            if not os.path.exists(os.path.join(path, r'{}'.format(
                    str(self.__class__.__name__)))):
                env_dir = os.path.join(path, r'{}'.format(str(self.__class__.__name__)))
                os.makedirs(os.path.join(env_dir, 'all_step_metrics'))

            np.savez(
                os.path.join(path, r'{}\all_step_metrics\{}.npz'.format(str(self.__class__.__name__), platform.name)),
                pr=platform.step_platform_revenue_list,
                cr=platform.step_completed_requests_list,
                orr=platform.step_order_response_rate_list,
                cancel_r=platform.step_cancel_requests_list,
                crr=platform.step_cancel_rate_list,
                md=platform.step_match_distance_list,
                pt=platform.step_pick_times_list,
                wt=platform.step_waiting_times_list
            )
        np.savez(
            os.path.join(path, r'{}\all_step_metrics\all_environment.npz'.format(str(self.__class__.__name__))),
            pr=self.step_market_revenue_list,
            cr=self.step_completed_requests_list,
            orr=self.step_order_response_rate_list,
            cancel_r=self.step_cancel_requests_list,
            crr=self.step_cancel_rate_list,
            md=self.step_match_distance_list,
            pt=self.step_pick_times_list,
            wt=self.step_waiting_times_list
        )


class RoadChangedEnvironment(Environment):
    """
    The RoadChangedEnvironment class represents a simulation environment where the roads can change. It extends the
    Environment class and adds additional properties and methods to handle changing roads.

    Attributes:
        where_change (list): A list indicating where changes occur.
        when_change (list): A list indicating when changes occur.
        can_drive (list): A list indicating where driving is possible.
        can_walk (list): A list indicating where walking is possible.
        road_network (RoadChangedRoadNetwork): The road network for this environment.
        control_driver_back_idle_num (int): A count of the number of drivers who have returned to idle state.
        control_driver_change_route_num (int): A count of the number of drivers who have changed their routes.
        control_driver_no_path_cancel_num (int): A count of the number of drivers who have cancelled due to lack of a path.
        control_driver_need_reposition_num (int): A count of the number of drivers who need to reposition.
        control_driver_change_destination_num (int): A count of the number of drivers who have changed their destination.
        control_passenger_additional_walk_num (int): A count of the number of passengers who have had additional walks.
        control_activate_passenger_request_num (int): A count of the number of passenger requests that have been activated.
        control_unactivated_passenger_request_num (int): A count of the number of passenger requests that have not been activated.
    """
    def __init__(
            self, city: str, platform_path: str, drive_speed: float, walk_speed: float, time_limit: int,
            save_path: str, where_change: list, when_change: list, can_drive: list, can_walk: list
    ):
        super().__init__(city, platform_path, drive_speed, walk_speed, time_limit, save_path)

        self.where_change = where_change
        self.when_change = when_change
        self.can_drive = can_drive
        self.can_walk = can_walk

        self.road_network = RoadChangedRoadNetwork(
            self.city, self.drive_speed, self.walk_speed, self.where_change, self.when_change, self.can_drive,
            self.can_walk
        )
        self.platforms = self.load_platform()
        self.pre_update_nodes()

        self.control_driver_back_idle_num = 0
        self.control_driver_change_route_num = 0
        self.control_driver_no_path_cancel_num = 0
        self.control_driver_need_reposition_num = 0
        self.control_driver_change_destination_num = 0
        self.control_passenger_additional_walk_num = 0
        self.control_activate_passenger_request_num = 0
        self.control_unactivated_passenger_request_num = 0

        self.save_path = save_path
        self.create_platform_save_dir(self.save_path)

    def load_platform(self):
        """
        Loads the platform operating settings from the YAML file specified in platform_path.

        Returns:
            list: A list of RoadChangedPlatform objects, each representing the settings of a platform.

        Note:
            The Platform class needs to be defined elsewhere and should take a dictionary of settings as parameters.
        """
        with open(self.platform_path, 'r') as f:
            platform_info = yaml.safe_load(f)
        return [RoadChangedPlatform(**p) for p in platform_info]

    def pre_update_nodes(self):
        """
        Updates the nodes for all drivers and passengers in all platforms within the environment.
        """
        [platform.drivers.update_nodes(self.road_network) for platform in self.platforms]
        [platform.passengers.update_dep_arr_nodes(self.road_network) for platform in self.platforms]

    def step(self):
        """
        Executes a single step in the simulation, including pre-dispatching for all platforms, collecting
        pre-statistics, dispatching and post-dispatching for all platforms, collecting post-statistics,
        resetting the step, and updating to the next time step.
        """
        [platform.pre_dispatch(self.time, self.road_network) for platform in self.platforms]
        self.pre_statistics()

        [platform.dispatch_post_dispatch(self.time, self.road_network) for platform in self.platforms]
        self.post_statistics()

        self.step_save_trajectories()
        self.step_reset()
        self.update_to_next_time()

    def pre_statistics(self):
        """
        Collects the statistics before dispatching for all platforms. This includes statistics about the
        number of waiting passengers, on-board passengers, passengers during pick-up and delivery, and
        the number of idle drivers, drivers during pick-up, delivery, and repositioning.
        """
        for platform in self.platforms:
            self.waiting_passenger_num += platform.waiting_passenger_num
            self.on_board_passenger_num += platform.on_board_passenger_num
            self.during_pick_passenger_num += platform.during_pick_passenger_num
            self.during_delivery_passenger_num += platform.during_delivery_passenger_num

            self.idle_driver_num += platform.idle_driver_num
            self.pick_driver_num += platform.pick_driver_num
            self.delivery_driver_num += platform.delivery_driver_num
            self.reposition_driver_num += platform.reposition_driver_num

    def update_to_next_time(self):
        """
        Updates the road impedance, increments the time step by 1, and updates the current road information
        based on the new time step.
        """
        self.update_road_impedance()
        self.time += 1
        self.road_network.update_current_road_info(self.time)

    def update_road_impedance(self):  # 更新道路阻抗
        """
        Updates the impedance of each route in the road network based on the number of drivers on each route.

        This function is typically called at each time step to reflect the change in road conditions.
        """
        route_driver_num = self.route_driver_counts()
        self.road_network.update_impedance(route_driver_num)

    def final_statistics(self):
        """
        Collects the final statistics after all steps have been performed for all platforms. This includes
        statistics about the market revenue, completed requests, order response rate, canceled requests,
        average match distance, pick-up times, waiting times, driver income, and driver orders. It also
        includes counts of various driver and passenger behaviors.
        """
        [platform.final_statistics() for platform in self.platforms]
        for platform in self.platforms:
            self.market_revenue += platform.platform_revenue
            self.completed_requests += platform.completed_requests
            self.order_response_rate += platform.order_response_rate
            self.cancel_requests += platform.cancel_requests
            self.cancel_rate += platform.cancel_rate
            self.avg_match_distance += platform.avg_match_distance
            self.avg_pick_times += platform.avg_pick_times
            self.avg_waiting_times += platform.avg_waiting_times
            self.avg_driver_income += platform.avg_driver_income
            self.avg_driver_order += platform.avg_driver_order

            self.control_driver_back_idle_num += platform.control_driver_back_idle_num
            self.control_driver_change_route_num += platform.control_driver_change_route_num
            self.control_driver_no_path_cancel_num += platform.control_driver_no_path_cancel_num
            self.control_driver_need_reposition_num += platform.control_driver_need_reposition_num
            self.control_driver_change_destination_num += platform.control_driver_change_destination_num
            self.control_passenger_additional_walk_num += platform.control_passenger_additional_walk_num
            self.control_activate_passenger_request_num += platform.control_activate_passenger_request_num
            self.control_unactivated_passenger_request_num += platform.control_unactivated_passenger_request_num

        self.order_response_rate = self.order_response_rate / len(self.platforms)
        self.cancel_rate = self.cancel_rate / len(self.platforms)
        self.avg_match_distance = self.avg_match_distance / len(self.platforms)
        self.avg_pick_times = self.avg_pick_times / len(self.platforms)
        self.avg_waiting_times = self.avg_waiting_times / len(self.platforms)
        self.avg_driver_income = self.avg_driver_income / len(self.platforms)
        self.avg_driver_order = self.avg_driver_order / len(self.platforms)

