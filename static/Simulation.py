# -*- coding: utf-8
# @Time : 2023/6/18 15:42
# @Author: Yinquan WANG
# @Email : 19114012@bjtu.edu.cn
# @File : Simulation.py
# @Project : 网约出行仿真
import os
import pickle
import logging
import datetime
import numpy as np
from tqdm import tqdm
from static.Environment import Environment, RoadChangedEnvironment


class Simulation:
    def __init__(
            self, name: str, city: str, platform_path: str, time_limit: int, drive_speed: float, walk_speed: float,
            road_changed: bool, where_change: list = None, when_change: list = None, can_drive: list = None,
            can_walk: list = None
    ):
        """
        Initializes the Simulation class.

        Parameters:
            name (str): The name of the simulation.
            city (str): The city where the simulation is conducted.
            platform_path (str): The path to the platform file.
            time_limit (int): The time limit for the simulation.
            drive_speed (float): The driving speed.
            walk_speed (float): The walking speed.
            road_changed (bool): If True, it indicates the road conditions have changed.
            where_change (list, optional): A list indicating the locations of changes.
                It should be provided when road_changed is True.
            when_change (list, optional): A list indicating the times when the changes occurred.
                It should be provided when road_changed is True.
            can_drive (list, optional): A list indicating whether driving is possible during the changes.
                It should be provided when road_changed is True.
            can_walk (list, optional): A list indicating whether walking is possible during the changes.
                It should be provided when road_changed is True.
        """
        self.name = name
        self.city = city
        self.platform_path = platform_path
        self.time_limit = time_limit
        self.drive_speed = drive_speed
        self.walk_speed = walk_speed
        self.road_changed = road_changed
        self.save_path = r'.\Results\{}'.format(self.name)

        self._create_simulate_path()
        self.logger = self._initialize_logger()
        if road_changed:
            self.where_change = where_change
            self.when_change = when_change
            self.can_drive = can_drive
            self.can_walk = can_walk
            self._check_input()

            self.main_env = RoadChangedEnvironment(
                self.city, self.platform_path, self.drive_speed, self.walk_speed, self.time_limit, self.save_path,
                self.where_change, self.when_change, self.can_drive, self.can_walk
            )
            self.compare_env = Environment(
                self.city, self.platform_path, self.drive_speed, self.walk_speed, self.time_limit, self.save_path
            )

            # statistics
            self.market_revenue_gap = None
            self.completed_requests_gap = None
            self.order_response_rate_gap = None
            self.cancel_requests_gap = None
            self.cancel_rate_gap = None
            self.avg_match_distance_gap = None
            self.avg_pick_times_gap = None
            self.avg_waiting_times_gap = None
            self.avg_driver_income_gap = None
            self.avg_driver_order_gap = None

            self.market_revenue = 0  # 市场收入
            self.completed_requests = 0  # 完成请求数
            self.order_response_rate = 0  # 订单响应率
            self.cancel_requests = 0  # 取消请求数
            self.cancel_rate = 0  # 订单取消率
            self.avg_match_distance = 0  # 平均匹配距离
            self.avg_pick_times = 0  # 平均接客时长
            self.avg_waiting_times = 0  # 平均等待时长
            self.avg_driver_income = 0  # 平均司机收入
            self.avg_driver_order = 0  # 平均司机订单数
        else:
            self.main_env = Environment(
                self.city, self.platform_path, self.drive_speed, self.walk_speed, self.time_limit, self.save_path
            )
            self.market_revenue = 0  # 市场收入
            self.completed_requests = 0  # 完成请求数
            self.order_response_rate = 0  # 订单响应率
            self.cancel_requests = 0  # 取消请求数
            self.cancel_rate = 0  # 订单取消率
            self.avg_match_distance = 0  # 平均匹配距离
            self.avg_pick_times = 0  # 平均接客时长
            self.avg_waiting_times = 0  # 平均等待时长
            self.avg_driver_income = 0  # 平均司机收入
            self.avg_driver_order = 0  # 平均司机订单数

        self.time_range = round(self.time_limit / 60, 2)
        self.platform_num = len(self.main_env.platforms)
        self.drivers_num = np.sum([platform.drivers.array.shape[0] for platform in self.main_env.platforms])
        self.passengers_num = np.sum([platform.passengers.array.shape[0] for platform in self.main_env.platforms])

        self._config_save()

    def _create_simulate_path(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _config_save(self):
        config = {
            'name': self.name, 'city': self.city, 'platform_path': self.platform_path, 'time_limit': self.time_limit,
            'drive_speed': self.drive_speed, 'walk_speed': self.walk_speed, 'road_changed': self.road_changed,
            'save_path': self.save_path, 'where_change': self.where_change, 'when_change': self.when_change,
            'can_drive': self.can_drive, 'can_walk': self.can_walk,
            'log_path': os.path.join(self.save_path, 'simulation.log')
        }
        config_path = os.path.join(self.save_path, 'simulate_config.pickle')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

    def _initialize_logger(self):
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(os.path.join(self.save_path, 'simulation.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    def _check_input(self):
        """
        Checks the input parameters for the Simulation class.

        Raises:
            ValueError: If road_changed is True and any of where_change, when_change, can_drive, can_walk is None.
            TypeError: If where_change, when_change, can_drive, can_walk are not of list type.
            ValueError: If where_change, when_change, can_drive, can_walk do not have equal lengths.
        """
        if self.road_changed:
            if self.where_change is None or self.when_change is None or self.can_drive is None or self.can_walk is None:
                raise ValueError(
                    "when road_changed is True, where_change, when_change, can_drive and can_walk should not be None")
            if not isinstance(self.where_change, list) or not isinstance(self.when_change, list) or not isinstance(
                    self.can_drive, list) or not isinstance(self.can_walk, list):
                raise TypeError("where_change, when_change, can_drive and can_walk should be list type")
            if not (len(self.where_change) == len(self.when_change) == len(self.can_drive) == len(self.can_walk)):
                raise ValueError("where_change, when_change, can_drive and can_walk should have equal length")

    def before_simulate(self):
        """
        Initializes the start time of the simulation.

        This method should be called before the simulation starts.
        """
        self.begin_time = datetime.datetime.now()
        self.begin_time_str = str(self.begin_time).split('.')[0]
        self.logger.info(f"The simulation name is: {self.name}.")
        self.logger.info('Simulation begins...')

    def after_simulate(self):
        """
        Initializes the end time of the simulation and calculates the duration.

        This method should be called after the simulation ends.
        """
        self.end_time = datetime.datetime.now()
        self.duration = (self.end_time - self.begin_time).total_seconds() / 60

        self.final_statistics()
        if self.road_changed:
            self.road_change_compared()
        self.logger.info('Simulation finishes...')
        self.logger_final_info()

        self.save_step_metrics()

    def simulate(self):
        """
        Runs the simulation for the given time limit.

        If road conditions have changed (road_changed is True), it runs both the main and comparison environments.
        Otherwise, it only runs the main environment.
        """
        self.before_simulate()
        if self.road_changed:
            for i in tqdm(range(self.time_limit)):
                self.main_env.step()
                self.compare_env.step()
            self.main_env.final_statistics()
            self.compare_env.final_statistics()
        else:
            for i in tqdm(range(self.time_limit)):
                self.main_env.step()
            self.main_env.final_statistics()
        self.after_simulate()

    def final_statistics(self):
        """
        Collects the final statistics from the main environment.

        The statistics include market revenue, completed requests, order response rate, cancelled requests,
        cancellation rate, average match distance, average pick-up times, average waiting times,
        average driver income, and average driver order number.
        """
        self.market_revenue = self.main_env.market_revenue
        self.completed_requests = self.main_env.completed_requests
        self.order_response_rate = self.main_env.order_response_rate
        self.cancel_requests = self.main_env.cancel_requests
        self.cancel_rate = self.main_env.cancel_rate
        self.avg_match_distance = self.main_env.avg_match_distance
        self.avg_pick_times = self.main_env.avg_pick_times
        self.avg_waiting_times = self.main_env.avg_waiting_times
        self.avg_driver_income = self.main_env.avg_driver_income
        self.avg_driver_order = self.main_env.avg_driver_order

    def road_change_compared(self):
        """
        Compares the final statistics between the main and comparison environments.

        The comparison is made for market revenue, completed requests, order response rate, cancelled requests,
        cancellation rate, average match distance, average pick-up times, average waiting times,
        average driver income, and average driver order number.
        """
        self.market_revenue_gap = self.main_env.market_revenue - self.compare_env.market_revenue
        self.completed_requests_gap = self.main_env.completed_requests - self.compare_env.completed_requests
        self.order_response_rate_gap = self.main_env.order_response_rate - self.compare_env.order_response_rate
        self.cancel_requests_gap = self.main_env.cancel_requests - self.compare_env.cancel_requests
        self.cancel_rate_gap = self.main_env.cancel_rate - self.compare_env.cancel_rate
        self.avg_match_distance_gap = self.main_env.avg_match_distance - self.compare_env.avg_match_distance
        self.avg_pick_times_gap = self.main_env.avg_pick_times - self.compare_env.avg_pick_times
        self.avg_waiting_times_gap = self.main_env.avg_waiting_times - self.compare_env.avg_waiting_times
        self.avg_driver_income_gap = self.main_env.avg_driver_income - self.compare_env.avg_driver_income
        self.avg_driver_order_gap = self.main_env.avg_driver_order - self.compare_env.avg_driver_order

    def logger_final_info(self):
        self.logger.info('=======================================Summary=======================================')
        self.logger.info('== Simulation Name: {:<11s} , Date : {:<10s},  City: {:<10s}    =='.format(
            self.name, str(self.begin_time_str), self.city
        ))
        self.logger.info('== Simulation duration (min): {:<8s}, Platforms num:   {:<2s}, Road change: {}     =='.format(
            str(round(self.duration, 2)), str(self.platform_num), str(self.road_changed)
        ))
        self.logger.info('== Time  Range  : 00.0 to {:<4s}, Vehicle supply: {:<8s} , Travel demand: {:<7s}  =='.format(
            str(self.time_range), str(self.drivers_num), str(self.passengers_num)
        ))
        self.logger.info('====================================Main Env=========================================')
        self.logger.info("== Total market revenue (CNY): {:<13.2f}, Completed  requests : {:<9.2f}      ==".format(
            self.market_revenue, self.completed_requests))
        self.logger.info("== Order responsive rate  (%): {:<13.2f}, Cancelled  requests : {:<9.2f}      ==".format(
            self.order_response_rate, self.cancel_requests))
        self.logger.info("== Cancellation    rates  (%): {:<13.2f}, Avg match distances (m): {:<6.2f}      ==".format(
            self.cancel_rate, self.avg_match_distance))
        self.logger.info("== Average pick-up times(min): {:<13.2f}, Avg waiting times(min): {:<7.2f}      ==".format(
            self.avg_pick_times, self.avg_waiting_times))
        self.logger.info("== Average driver income(CNY): {:<13.2f}, Avg  driver  orders : {:<9.2f}      ==".format(
            self.avg_driver_income, self.avg_driver_order))
        if self.road_changed:
            self.logger.info('================================Environment Gap======================================')
            self.logger.info("== Total market revenue GAP(CNY): {:<10.2f}, Completed  requests : {:<9.2f}      ==".format(
                self.market_revenue_gap, self.completed_requests_gap))
            self.logger.info("== Order response rate GAP (%): {:<12.2f}, Cancellation request: {:<9.2f}      ==".format(
                self.order_response_rate_gap, self.cancel_requests_gap))
            self.logger.info("== Cancellation  rate  GAP (%): {:<12.2f}, Avg match distances (m): {:<6.2f}      ==".format(
                self.cancel_rate_gap, self.avg_match_distance_gap))
            self.logger.info("== Avg  pick-up times GAP(min): {:<12.2f}, Avg waiting times (min): {:<6.2f}      ==".format(
                self.avg_pick_times_gap, self.avg_waiting_times_gap))
            self.logger.info("== Avg  driver income GAP(CNY): {:<12.2f}, Avg  driver  orders : {:<9.2f}      ==".format(
                self.avg_driver_income_gap, self.avg_driver_order_gap))
            self.logger.info('================================Compared Environment=================================')
            self.logger.info("== Total  market revenue(CNY): {:<13.2f}, Completed  requests : {:<9.2f}      ==".format(
                self.compare_env.market_revenue, self.compare_env.completed_requests))
            self.logger.info("== Order responsive rate  (%): {:<13.2f}, Cancelled  requests : {:<9.2f}      ==".format(
                self.compare_env.order_response_rate, self.compare_env.cancel_requests))
            self.logger.info("== Cancellation    rates  (%): {:<13.2f}, Avg match distances (m): {:<6.2f}      ==".format(
                self.compare_env.cancel_rate, self.compare_env.avg_match_distance))
            self.logger.info("== Average pick-up times(min): {:<13.2f}, Avg waiting times (min): {:<6.2f}      ==".format(
                self.compare_env.avg_pick_times, self.compare_env.avg_waiting_times))
            self.logger.info("== Average driver income(CNY): {:<13.2f}, Avg  driver  orders : {:<9.2f}      ==".format(
                self.compare_env.avg_driver_income, self.compare_env.avg_driver_order))
            self.logger.info('================================Road Change Influences===============================')
            self.logger.info("== Drivers return to the idle: {:<8.2f}, Drivers  have to  change routes: {:<8.2f} ==".format(
                self.main_env.control_driver_back_idle_num, self.main_env.control_driver_change_route_num))
            self.logger.info("== Drivers no path to  cancel: {:<8.2f}, Drivers reposition  out control: {:<8.2f} ==".format(
                self.main_env.control_driver_no_path_cancel_num, self.main_env.control_driver_need_reposition_num))
            self.logger.info("== Drivers change destination: {:<8.2f}, Passengers with additional walk: {:<8.2f} ==".format(
                self.main_env.control_driver_change_destination_num,
                self.main_env.control_passenger_additional_walk_num))
            self.logger.info("== Activate passengers cancel: {:<8.2f}, Unactivated  passengers  cancel: {:<8.2f} ==".format(
                self.main_env.control_activate_passenger_request_num,
                self.main_env.control_unactivated_passenger_request_num))
        self.logger.info('=====================================================================================')

    def summary(self):
        print('=======================================Summary=======================================')
        print('== Simulation Name: {:<11s} , Date : {:<10s},  City: {:<10s}    =='.format(
            self.name, str(self.begin_time_str), self.city
        ))
        print('== Simulation duration (min): {:<8s}, Platforms num:   {:<2s}, Road change: {}    =='.format(
            str(round(self.duration, 2)), str(self.platform_num), str(self.road_changed)
        ))
        print('== Time  Range  : 00.0 to {:<4s}, Vehicle supply: {:<8s} , Travel demand: {:<7s}  =='.format(
            str(self.time_range), str(self.drivers_num), str(self.passengers_num)
        ))
        print('====================================Main Env=========================================')
        print("== Total market revenue (CNY): {:<13.2f}, Completed  requests : {:<9.2f}      ==".format(
            self.market_revenue, self.completed_requests))
        print("== Order responsive rate  (%): {:<13.2f}, Cancelled  requests : {:<9.2f}      ==".format(
            self.order_response_rate, self.cancel_requests))
        print("== Cancellation    rates  (%): {:<13.2f}, Avg match distances (m): {:<6.2f}      ==".format(
            self.cancel_rate, self.avg_match_distance))
        print("== Average pick-up times(min): {:<13.2f}, Avg waiting times(min): {:<7.2f}      ==".format(
            self.avg_pick_times, self.avg_waiting_times))
        print("== Average driver income(CNY): {:<13.2f}, Avg  driver  orders : {:<9.2f}      ==".format(
            self.avg_driver_income, self.avg_driver_order))
        if self.road_changed:
            print('================================Environment Gap======================================')
            print("== Total market revenue GAP(CNY): {:<10.2f}, Completed  requests : {:<9.2f}      ==".format(
                self.market_revenue_gap, self.completed_requests_gap))
            print("== Order response rate GAP (%): {:<12.2f}, Cancellation request: {:<9.2f}      ==".format(
                self.order_response_rate_gap, self.cancel_requests_gap))
            print("== Cancellation  rate  GAP (%): {:<12.2f}, Avg match distances (m): {:<6.2f}      ==".format(
                self.cancel_rate_gap, self.avg_match_distance_gap))
            print("== Avg  pick-up times GAP(min): {:<12.2f}, Avg waiting times (min): {:<6.2f}      ==".format(
                self.avg_pick_times_gap, self.avg_waiting_times_gap))
            print("== Avg  driver income GAP(CNY): {:<12.2f}, Avg  driver  orders : {:<9.2f}      ==".format(
                self.avg_driver_income_gap, self.avg_driver_order_gap))
            print('================================Compared Environment=================================')
            print("== Total  market revenue(CNY): {:<13.2f}, Completed  requests : {:<9.2f}      ==".format(
                self.compare_env.market_revenue, self.compare_env.completed_requests))
            print("== Order responsive rate  (%): {:<13.2f}, Cancelled  requests : {:<9.2f}      ==".format(
                self.compare_env.order_response_rate, self.compare_env.cancel_requests))
            print("== Cancellation    rates  (%): {:<13.2f}, Avg match distances (m): {:<6.2f}      ==".format(
                self.compare_env.cancel_rate, self.compare_env.avg_match_distance))
            print("== Average pick-up times(min): {:<13.2f}, Avg waiting times (min): {:<6.2f}      ==".format(
                self.compare_env.avg_pick_times, self.compare_env.avg_waiting_times))
            print("== Average driver income(CNY): {:<13.2f}, Avg  driver  orders : {:<9.2f}      ==".format(
                self.compare_env.avg_driver_income, self.compare_env.avg_driver_order))
            print('================================Road Change Influences===============================')
            print("== Drivers return to the idle: {:<8.2f}, Drivers  have to  change routes: {:<8.2f} ==".format(
                self.main_env.control_driver_back_idle_num, self.main_env.control_driver_change_route_num))
            print("== Drivers no path to  cancel: {:<8.2f}, Drivers reposition  out control: {:<8.2f} ==".format(
                self.main_env.control_driver_no_path_cancel_num, self.main_env.control_driver_need_reposition_num))
            print("== Drivers change destination: {:<8.2f}, Passengers with additional walk: {:<8.2f} ==".format(
                self.main_env.control_driver_change_destination_num, self.main_env.control_passenger_additional_walk_num))
            print("== Activate passengers cancel: {:<8.2f}, Unactivated  passengers  cancel: {:<8.2f} ==".format(
                self.main_env.control_activate_passenger_request_num,
                self.main_env.control_unactivated_passenger_request_num))
        print('=====================================================================================')

    def save_step_metrics(self):
        """
        Save the step metrics for the main and comparison environments (if road conditions have changed).

        This function should be called after the simulation to persist the step metrics.
        """
        self.main_env.save_platform_step_metrics(self.save_path)
        if self.road_changed:
            self.compare_env.save_platform_step_metrics(self.save_path)

    def results_analysis(self):
        """
        Analyzes the results of the simulation and plots various metrics.

        This function generates a multi-plot figure with each subplot showing a different metric over time.
        Metrics include 'Market revenue', 'Completed requests', 'Order response rate', 'Cancel requests',
        'Cancel rate', 'Match distance', 'Pick-up times', and 'Waiting times'.
        If the road conditions have changed during the simulation, the metrics for the comparison environment
        are also plotted for comparison. Control periods are represented by shaded areas on the plots.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        plt.rcParams['font.size'] = 14

        main_env_path = os.path.join(self.save_path, self.main_env.__class__.__name__)
        main_env_data = self.load_results(main_env_path)
        main_env_summary = main_env_data['all_environment']

        fig, axes = plt.subplots(len(main_env_summary.keys()), 1, figsize=(10, 10))

        main_env_lines = []
        title_name = [
            'Market revenue', 'Completed requests', 'Order response rate', 'Cancel requests',
            'Cancel rate', 'Match distance', 'Pick-up times', 'Waiting times'
        ]
        y_label = ['Revenue $(CNY)$', 'Num', 'Percent $(\%)$', 'Num', 'Percent $(\%)$',
                   'Distance $(m)$', 'Duration $(min)$', 'Duration $(min)$']
        for i, (ax, (key, values)) in enumerate(zip(axes, main_env_summary.items())):
            line, = ax.plot(values, color='blue', label='Main env', mfc='w', alpha=0.8)
            ax.set_ylabel(y_label[i], fontdict={'size': 12})
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
            ax.annotate(title_name[i], (0.05, 0.7), xycoords='axes fraction')
            if i < len(main_env_summary.keys()) - 1:
                ax.tick_params(labelbottom=False)
            main_env_lines.append(line)

            # 添加阴影
            if self.road_changed and self.when_change:
                for change_interval in self.when_change:
                    ax.axvspan(change_interval[0], change_interval[1], facecolor='pink', alpha=0.5)

        if self.road_changed:
            compare_env_lines = []
            compare_env_path = os.path.join(self.save_path, self.compare_env.__class__.__name__)
            compare_env_data = self.load_results(compare_env_path)
            compare_env_summary = compare_env_data['all_environment']
            for ax, (key, values) in zip(axes, compare_env_summary.items()):
                line, = ax.plot(values, color='red', label='Compare env', mfc='w', alpha=0.8)
                compare_env_lines.append(line)
            control_patch = Patch(facecolor='pink', label='Control period',
                                  alpha=0.5)  # create a legend entry for control period
            plt.legend([main_env_lines[0], compare_env_lines[0], control_patch],
                       ['Main env', 'Compare env', 'Control period'],
                       loc='lower center', bbox_to_anchor=(0.5, -1.2), ncol=3)
        else:
            plt.legend([main_env_lines[0]], ['Main env'],
                       loc='lower center', bbox_to_anchor=(0.5, -1.2), ncol=3)
        axes[-1].set_xlabel('Times (min)')
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.show(block=True)

    def load_results(self, path):
        """
        Load the saved results from a given directory.

        Parameters:
            path (str): Path to the directory where the results are stored.
        Returns:
            save_dict (dict): Dictionary containing the results for each platform.
        """
        metrics_path = os.path.join(path, 'all_step_metrics')
        metrics_names = os.listdir(metrics_path)
        save_dict = dict()
        for name in metrics_names:
            save_dict[name.split('.')[0]] = dict(np.load(os.path.join(metrics_path, name)))
        return save_dict
