import os
import pickle
import numpy as np
from static.Simulation import Simulation


class SimulationReplay:
    """
    This class is used to replay the simulation based on the configuration.

    Parameters
    ----------
    config_path : str
        The path of the configuration file.
    """
    def __init__(self, config_path):
        """
        Constructs all the necessary attributes for the SimulationReplay object.

        Parameters
        ----------
        config_path : str
            The path of the configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()

        self.simulation = self._create_origin_simulation()

    def _create_origin_simulation(self):
        """
        Creates the origin simulation based on the configuration.

        Returns
        -------
        Simulation
            The created origin simulation object.
        """
        return Simulation(
            self.config['name'], self.config['city'], self.config['platform_path'], self.config['time_limit'],
            self.config['drive_speed'], self.config['walk_speed'], self.config['road_changed'],
            self.config['where_change'], self.config['when_change'], self.config['can_drive'],
            self.config['can_walk'],
        )

    def load_logger(self):
        """
        Loads the logs from the log file and print them.
        """
        log_path = self.config['log_path']
        with open(log_path, 'r') as file:
            logs = file.read()
        print(logs)

    def _load_config(self):
        """
        Loads the configuration from the configuration file.

        Returns
        -------
        dict
            The loaded configuration.
        """
        with open(self.config_path, 'rb') as f:
            config = pickle.load(f)
        return config

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

        main_env_path = os.path.join(self.config['save_path'], 'RoadChangedEnvironment')
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
            if self.config['road_changed'] and self.config['when_change']:
                for change_interval in self.config['when_change']:
                    ax.axvspan(change_interval[0], change_interval[1], facecolor='pink', alpha=0.5)

        if self.config['road_changed']:
            compare_env_lines = []
            compare_env_path = os.path.join(self.config['save_path'], 'Environment')
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

    def visualization(self):
        """
        Creates an interactive visualization of a simulation.

        This method generates a plot of a road network, with points representing vehicles and passengers
        in various states. The plot can be paused and resumed by clicking a button, and the simulation
        speed can be adjusted with a slider.

        The road network is gray, with transparency to allow visibility of the points. Vehicles are
        represented with filled markers, with color and shape indicating their status and platform.
        Passengers are represented with unfilled markers, again with color and shape indicating their
        status and platform.

        Vehicles statuses are as follows: 0: "Idle Vehicles", 1: "Pick Vehicles", 2: "Delivery Vehicles",
        3: "Reposition Vehicles".

        Passengers statuses are as follows: 1: "Walk Passengers", 2: "Wait Passengers", 3: "Wait Pick Passengers",
        4: "Wait Delivery Passengers".

        Note:
            This function relies on data loaded from the `load_frame_data` function and the `main_env` object,
            which should be properly initialized before calling this function.
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.widgets import Button, Slider

        plt.rcParams['font.size'] = 14

        markers = ['o', 's', 'v', '^', '>', '<', '8', 'p', 'h', 'H', 'D', 'd', 'P', 'X']
        vehicle_status = {0: "Idle Vehicles", 1: "Pick Vehicles", 2: "Delivery Vehicles", 3: "Reposition Vehicles"}
        passenger_status = {1: "Walk Passengers", 2: "Wait Passengers", 3: "Wait Pick Passengers",
                            4: "Wait Delivery Passengers"}

        vehicle_colors = {0: "black", 1: "blue", 2: "red", 3: "yellow"}
        passenger_colors = {1: "black", 2: "blue", 3: "red", 4: "yellow"}

        fig, ax = plt.subplots(figsize=(16, 15))
        plt.subplots_adjust(bottom=0.3)

        # 加载数据
        main_env_first_drivers, main_env_first_passengers, main_env_first_metrics = \
            self.load_frame_data(self.simulation.main_env, 0)

        road_index = self.simulation.main_env.road_network.find_interval(0)
        road = self.simulation.main_env.road_network.gdf_edges if road_index == -1\
            else self.simulation.main_env.road_network.changed_edge_list[road_index]
        road.plot(ax=ax, alpha=0.1, color='gray', zorder=1)  # 设置透明度

        driver_scatters = []
        passenger_scatters = []
        for i in range(len(self.simulation.main_env.platforms)):
            for status, color in zip(vehicle_status, vehicle_colors):
                vehicles = main_env_first_drivers[i][np.where(main_env_first_drivers[i][:, 2] == status)]
                if i == 0:
                    scatter = ax.scatter(vehicles[:, 0], vehicles[:, 1], c=vehicle_colors[status], marker=markers[i],
                                         s=40, zorder=2, label=vehicle_status[status], alpha=0.5)
                else:
                    scatter = ax.scatter(vehicles[:, 0], vehicles[:, 1], c=vehicle_colors[status], marker=markers[i],
                                         s=40, zorder=2, linewidths=20, alpha=0.5)
                driver_scatters.append(scatter)

            for status, color in zip(passenger_status, passenger_colors):
                passengers = main_env_first_passengers[i][np.where(main_env_first_passengers[i][:, 2] == status)]
                if i == 0:
                    scatter = ax.scatter(passengers[:, 0], passengers[:, 1], facecolors='none',
                                         edgecolors=passenger_colors[status], marker=markers[i], s=40, zorder=2,
                                         label=passenger_status[status], alpha=0.5)
                else:
                    scatter = ax.scatter(passengers[:, 0], passengers[:, 1], facecolors='none',
                                         edgecolors=passenger_colors[status], marker=markers[i], s=40, zorder=2,
                                         linewidths=20, alpha=0.5)
                passenger_scatters.append(scatter)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        metrics_text = '\n\n'.join(f'{k.replace("_", " ").title()}: {np.round(v, 2)}'
                                 for k, v in main_env_first_metrics.item().items())

        ax.text(1.02, 0.695, metrics_text, transform=ax.transAxes, va='top', ha='left', fontsize=14)
        self.is_running = True

        def animate(env_time):
            main_env_drivers, main_env_passengers, main_env_metrics = self.load_frame_data(self.simulation.main_env,
                                                                                           env_time)
            ax.clear()

            metrics_text = '\n\n'.join(f'{k.replace("_", " ").title()}: {np.round(v, 2)}'
                                       for k, v in main_env_metrics.item().items())
            ax.text(1.02, 0.695, metrics_text, transform=ax.transAxes, va='top', ha='left', fontsize=14)

            road_index = self.simulation.main_env.road_network.find_interval(env_time)
            road = self.simulation.main_env.road_network.gdf_edges if road_index == -1 \
                else self.simulation.main_env.road_network.changed_edge_list[road_index]
            road.plot(ax=ax, alpha=0.1, color='gray', zorder=1)

            # 创建空的列表来存储新的散点图
            new_driver_scatters = []
            new_passenger_scatters = []

            for j in range(len(self.simulation.main_env.platforms)):
                if len(main_env_drivers[j]) > 0:  # 检查driver数据是否存在
                    for status, color in zip(vehicle_status, vehicle_colors):
                        vehicles = main_env_drivers[j][np.where(main_env_drivers[j][:, 2] == status)]
                        if i == 0:
                            scatter = ax.scatter(vehicles[:, 0], vehicles[:, 1], c=vehicle_colors[status],
                                                 marker=markers[j], s=40, zorder=2, label=vehicle_status[status],
                                                 alpha=0.5)
                        else:
                            scatter = ax.scatter(vehicles[:, 0], vehicles[:, 1], c=vehicle_colors[status],
                                                 marker=markers[j], s=40, zorder=2, linewidths=20, alpha=0.5)
                        new_driver_scatters.append(scatter)

                if len(main_env_passengers[j]) > 0:  # 检查passenger数据是否存在
                    for status, color in zip(passenger_status, passenger_colors):
                        passengers = main_env_passengers[j][np.where(main_env_passengers[j][:, 2] == status)]
                        if i == 0:
                            scatter = ax.scatter(passengers[:, 0], passengers[:, 1], facecolors='none',
                                                 edgecolors=passenger_colors[status], marker=markers[j], s=40,
                                                 zorder=2, label=passenger_status[status], alpha=0.5)
                        else:
                            scatter = ax.scatter(passengers[:, 0], passengers[:, 1], facecolors='none',
                                                 edgecolors=passenger_colors[status], marker=markers[j], s=40,
                                                 zorder=2, linewidths=20, alpha=0.5)
                        new_passenger_scatters.append(scatter)

            hours, remainder = divmod(env_time, 60)
            minutes = remainder
            # 用格式化字符串显示时间
            time_str = f'{hours:02}:{minutes:02}'
            ax.set_title(f'Time: {time_str}')  # 添加图名
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            return new_driver_scatters + new_passenger_scatters

        def onClick(event):
            if self.is_running:
                ani.event_source.stop()
                self.is_running = False
            else:
                ani.event_source.start()
                self.is_running = True

        def onSliderUpdate(val):
            ani.event_source.interval = 1000 / val  # 更新帧时间，而不是帧速率

        axpause = plt.axes([0.85, 0.12, 0.1, 0.075])
        button = Button(axpause, 'Pause')
        button.on_clicked(onClick)

        axcolor = 'lightgoldenrodyellow'
        axfreq = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
        speed_slider = Slider(axfreq, 'Speed (x normal)', 1/3, 2.0, valinit=1.0)
        speed_slider.on_changed(onSliderUpdate)

        ani = animation.FuncAnimation(
            fig, animate, frames=self.config['time_limit'], interval=1000, blit=False)
        plt.show(block=True)
        # ani.save('animation.gif', writer='imagemagick')

    def load_frame_data(self, env, env_time):
        """
        Load the frame data for drivers and passengers at a given time in the simulation.

        Parameters:
            env (Environment object) : The environment object, which contains the saved path and platforms.
            env_time (int) : The time at which to load the frame data.

        Returns:
            main_env_drivers, main_env_passengers: Two lists containing loaded frame data for drivers and passengers,
            respectively.
        """
        main_env_drivers = []
        main_env_passengers = []
        load_path = os.path.join(env.save_path, env.__class__.__name__)
        for i, platform in enumerate(env.platforms):
            platform_path = os.path.join(load_path, platform.name)
            driver_data = np.load(
                os.path.join(platform_path, r'drivers\drivers_time_{}.npy'.format(env_time)), allow_pickle=True)
            passenger_data = np.load(
                os.path.join(platform_path, r'passengers\passengers_time_{}.npy'.format(env_time)), allow_pickle=True)
            main_env_drivers.append(driver_data)
            main_env_passengers.append(passenger_data)
        env_metrics = np.load(os.path.join(load_path, r'single_step_metrics\metrics_time_{}.npy'.format(env_time)),
                              allow_pickle=True)
        return main_env_drivers, main_env_passengers, env_metrics
