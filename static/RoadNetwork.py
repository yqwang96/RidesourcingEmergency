# @Time : 2023/5/14 12:04 
# @Author : Yinquan Wang<19114012@bjtu.edu.cn>
# @File : RoadNetwork.py 
# @Function:
import pandana
import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd

import warnings

warnings.filterwarnings('ignore')


RADIUS = 6371
ALPHA = 0.15
BETA = 4
SPEED_DISCOUNT = 0.8
LANES_MAP_DICT = {
    'unclassified': 1, 'tertiary': 2, 'tertiary_link': 2, 'primary': 3, 'primary_link': 3, 'residential': 1,
    'secondary': 3, 'secondary_link': 3, 'trunk': 3, 'trunk_link': 3, 'motorway': 4, 'motorway_link': 4,
    'living_street': 1, 'road': 1
}
SPEED_MAP_DICT = {
    'unclassified': 20, 'tertiary': 60, 'tertiary_link': 40, 'primary': 60, 'primary_link': 40, 'residential': 20,
    'secondary': 60, 'secondary_link': 50, 'trunk': 60, 'trunk_link': 40, 'motorway': 100, 'motorway_link': 80,
    'living_street': 20, 'road': 20
}
CAPACITY_MAP_DICT = {
    'unclassified': 100, 'tertiary': 500, 'tertiary_link': 500, 'primary': 900, 'primary_link': 900, 'residential': 100,
    'secondary': 900, 'secondary_link': 900, 'trunk': 500, 'trunk_link': 500, 'motorway': 1800, 'motorway_link': 1800,
    'living_street': 100, 'road': 100
}


class RoadNetwork:
    def __init__(self, city: str, drive_speed: float, walk_speed: float):
        """
        Initialize a RoadNetwork instance with given city, drive speed and walk speed.

        Parameters:
            city (str): The city where the road network is located.
            drive_speed (float): The speed of driving in the road network.
            walk_speed (float): The speed of walking in the road network.

        Raises:
            FileNotFoundError: If the city is not supported.
        """
        self.city = city
        self.drive_speed = drive_speed
        self.walk_speed = walk_speed

        if city == 'beijing':
            self._road_network = nx.read_gpickle(r'E:\simulate_env\beijing_driveable1.gpickle')
        else:
            raise FileNotFoundError('The environment do not have the data of {}'.format(self.city))

        self.gdf_nodes, self.gdf_edges = ox.graph_to_gdfs(self._road_network)
        self.gdf_edges.reset_index(inplace=True)
        self.gdf_edges.reset_index(inplace=True)
        self.gdf_nodes['index'] = range(self.gdf_nodes.shape[0])

        map_dict = self.gdf_nodes['index'].to_dict()
        self.gdf_nodes.set_index('index', drop=False, inplace=True)
        self.gdf_edges['u'] = self.gdf_edges['u'].map(map_dict)
        self.gdf_edges['v'] = self.gdf_edges['v'].map(map_dict)

        # BPR函数动态计算阻抗
        self.gdf_edges['highway'] = self.gdf_edges['highway'].apply(lambda x: x[0] if type(x) is list else x)
        self.gdf_edges['lanes'] = self.gdf_edges['highway'].map(LANES_MAP_DICT)
        self.gdf_edges['free_speed'] = self.gdf_edges['highway'].map(SPEED_MAP_DICT) * SPEED_DISCOUNT
        self.gdf_edges['free_time'] = self.gdf_edges['length'] / 1000 / self.gdf_edges['free_speed'] * 60
        self.gdf_edges['capacity'] = self.gdf_edges['highway'].map(CAPACITY_MAP_DICT) * self.gdf_edges['lanes']
        self.gdf_edges['cars'] = 0
        self.gdf_edges['times'] = self.gdf_edges['free_time'] *\
                                  (1 + ALPHA * ((self.gdf_edges['cars'] / self.gdf_edges['capacity']) ** BETA))

        self.pandana_net = pandana.Network(
            self.gdf_nodes["x"], self.gdf_nodes["y"], self.gdf_edges["u"], self.gdf_edges["v"],
            self.gdf_edges[['length', 'times']], twoway=False
        )

        self.min_x, self.min_y = self.gdf_nodes[['x', 'y']].min()
        self.max_x, self.max_y = self.gdf_nodes[['x', 'y']].max()

        self.current_road = self.pandana_net
        self.current_nodes = self.gdf_nodes
        self.current_edges = self.gdf_edges

    @property
    def board(self):
        """
        Returns the bounding box of the road network.

        Returns:
            tuple: A tuple of four floats, (min_x, min_y, max_x, max_y), representing the coordinates of the bounding box.
        """
        return self.min_x, self.min_y, self.max_x, self.max_y

    @property
    def nodes(self):
        """
        Returns a list of all node indices in the road network.

        Returns:
            list: A list of node indices.
        """
        return self.current_nodes.index.tolist()

    @property
    def routes(self):
        """
        Returns a list of all route indices in the road network.

        Returns:
            list: A list of route indices.
        """
        return self.current_edges.index.tolist()

    def get_trajectories(self, dep_nodes, arr_nodes, dep_coords, arr_coords, dynamic_impedance):
        """
        This function calculates the shortest trajectories and routes for a given set of departure and arrival nodes.

        Parameters:
            dep_nodes (list): A list of departure nodes.
            arr_nodes (list): A list of arrival nodes.
            dep_coords (np.ndarray): A list of departure coordinates. Each element in the list should be a tuple of (x, y).
            arr_coords (np.ndarray): A list of arrival coordinates. Each element in the list should be a tuple of (x, y).
            dynamic_impedance (bool): If True, use 'times' as impedance, otherwise use 'length'.

        Returns:
            all_all_trajectories (list of list): A nested list of all trajectories. Each trajectory is a list of
            coordinates.
            all_all_routes (list of list): A nested list of all routes. Each route is a list of edge indices.

        Note:
            This function uses the Pandana library to compute the shortest paths, which uses the Dijkstra's algorithm
            under the hood. If dynamic_impedance is True, the impedance of each edge is calculated based on the travel
            time, otherwise the impedance is based on the edge length.

            This function also handles the situation where the departure and arrival nodes are the same. In this case,
            the trajectory is simply the coordinates of the node itself.
        """
        imp_name = 'times' if dynamic_impedance else 'length'
        dep_coords = dep_coords.tolist()
        arr_coords = arr_coords.tolist()
        routes = self.current_road.shortest_paths(dep_nodes, arr_nodes, imp_name=imp_name)  # 这个需要改
        link_origin = []
        end_origin = []
        no = []
        passenger = []
        all_all_trajectories = [[] for _ in routes]
        all_all_routes = [[] for _ in routes]
        not_at_same_node = []  # the indexes of passengers and drivers who are not on the same node
        for index, route in enumerate(routes):
            route = list(route)
            if len(route) == 1:
                all_all_trajectories[index] = [dep_coords[index]]
                all_all_routes[index] = [-1]
            else:
                not_at_same_node.append(index)
                link_origin += route[:-1]
                end_origin += route[1:]
                no += list(range(len(route) - 1))
                passenger += [index] * (len(route) - 1)

        if len(not_at_same_node) != 0:
            links_shp = pd.merge(
                pd.DataFrame({
                    'u': link_origin,
                    'v': end_origin,
                    'key': [0] * len(link_origin),
                    'no': no, 'passenger': passenger
                }), self.current_edges, left_on=['u', 'v', 'key'], right_on=['u', 'v', 'key'], how='left'
            )

            if imp_name == 'length':
                links_shp['sum_length'] = links_shp.groupby('passenger')['length'].cumsum()
                links_shp['on_label'] = links_shp['sum_length'] // self.drive_speed
                links_shp_sum = gpd.GeoDataFrame(links_shp).dissolve(
                    by='passenger', aggfunc={'length': 'sum', 'passenger': 'mean'}
                )
                links_shp_sum['num_points'] = (links_shp_sum['length'] // self.drive_speed).astype(int)
            elif imp_name == 'times':
                links_shp['sum_times'] = links_shp.groupby('passenger')['times'].cumsum()
                links_shp['on_label'] = links_shp['sum_times'].astype(int)
                links_shp_sum = gpd.GeoDataFrame(links_shp).dissolve(
                    by='passenger', aggfunc={'times': 'sum', 'passenger': 'mean'}
                )
                links_shp_sum['num_points'] = links_shp_sum['times'].astype(int)
            else:
                raise NotImplementedError

            passenger_label_max = links_shp.groupby('passenger').agg({'on_label': 'max'})['on_label'].to_dict()
            road = pd.merge(
                links_shp[['index', 'on_label', 'passenger']].drop_duplicates(subset=['on_label', 'passenger'],
                                                                              keep='last'),
                pd.DataFrame([{'passenger': key, 'on_label': i} for key in passenger_label_max.keys()
                              for i in range(int(passenger_label_max[key] + 1))]),
                left_on=['passenger', 'on_label'], right_on=['passenger', 'on_label'], how='right'
            )

            road['index'] = road.groupby('passenger', group_keys=False)['index'].apply(lambda x: x.fillna(method='ffill'))
            road['index'] = road.groupby('passenger', group_keys=False)['index'].apply(lambda x: x.fillna(method='bfill'))
            road['index'] = road['index'].apply(int)
            all_road = [road[road['passenger'] == p]['index'].tolist() for p in list(road['passenger'].unique())]

            links_shp_sum['inter_position'] = links_shp_sum.apply(lambda x: [x['geometry'].interpolate(
                i / x['num_points'], normalized=True).coords[0] for i in range(1, x['num_points'] + 1)], axis=1)

            trajectories = links_shp_sum['inter_position'].values

            all_trajectories = [trajectory + [arr_coords[index]] for index, trajectory in enumerate(trajectories)]
            for index, od_pair in enumerate(not_at_same_node):
                all_all_trajectories[od_pair] = all_trajectories[index]
                all_all_routes[od_pair] = all_road[index]
        return all_all_trajectories, all_all_routes

    def shortest_path_lengths(self, dep_nodes, arr_nodes, imp_name):
        """
        Calculate the shortest path lengths between departure nodes and arrival nodes.

        Parameters:
            dep_nodes (list): The departure nodes.
            arr_nodes (list): The arrival nodes.
            imp_name (str): The name of the impedance (distance, time, etc.)

        Returns:
            np.ndarray: An array of shortest path lengths.
        """
        return self.current_road.shortest_path_lengths(dep_nodes, arr_nodes, imp_name)

    def get_coord_node(self, coordinates):
        """
        Get the nearest nodes to the given coordinates.

        Parameters:
            coordinates (np.ndarray): The coordinates.

        Returns:
            np.ndarray: The array of nearest node indices.
        """
        if coordinates.shape[0] != 0:
            return self.current_road.get_node_ids(coordinates[:, 0], coordinates[:, 1])

    def get_node_coordinates(self, nodes):
        if nodes.shape[0] != 0:
            return self.current_nodes.loc[nodes, ['x', 'y']].values

    def calculate_walk_distance_coord(self, coordinates, nearest_node):
        """
        Calculate the walking distance between the given coordinates and their nearest nodes.

        Parameters:
            coordinates (np.ndarray): The coordinates.
            nearest_node (np.ndarray): The indices of nearest nodes.

        Returns:
            tuple: A tuple of two arrays, (walk_distance, walk_coordinates), representing walking distances and their
            coordinates.
        """
        walk_distance, walk_coordinates = self.manhattan_distance_road(coordinates, nearest_node)
        return walk_distance, walk_coordinates

    def manhattan_distance_road(self, coord, node):
        """
        Calculate the Manhattan distances between the given coordinates and their nodes on the road network.

        Parameters:
            coord (np.ndarray): The coordinates.
            node (np.ndarray): The nodes.

        Returns:
            tuple: A tuple of two arrays, (walk_distance, walk_coordinates), representing walking distances and their
            coordinates.
        """
        node_coord = self.current_nodes.loc[node, ['x', 'y']].values
        mid_coord = np.hstack((coord[:, [0]], node_coord[:, [1]]))
        walk_distance = self.haversine_distance(coord, mid_coord) + self.haversine_distance(mid_coord, node_coord)

        first_trajectories = self.interpolation_coordinates(coord, mid_coord, 1, self.walk_speed)
        second_trajectories = self.interpolation_coordinates(mid_coord, node_coord, 0, self.walk_speed)

        walk_coordinates = [i + j for i, j in zip(first_trajectories, second_trajectories)]
        return walk_distance, walk_coordinates

    def haversine_distance(self, coord1, coord2):
        """
        Calculate the haversine distance between two sets of coordinates.

        Parameters:
            coord1 (np.ndarray): The first set of coordinates.
            coord2 (np.ndarray): The second set of coordinates.

        Returns:
            np.ndarray: An array of distances between coord1 and coord2.
        """
        coord1, coord2 = np.radians([coord1, coord2])

        dlat = coord2[:, 1] - coord1[:, 1]
        dlon = coord2[:, 0] - coord1[:, 0]

        a = np.sin(dlat / 2) ** 2 + np.cos(coord1[:, 1]) * np.cos(coord2[:, 1]) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return RADIUS * c * 1e3

    def interpolation_coordinates(self, coord1, coord2, direction, speed):
        """
        Generate interpolated coordinates between coord1 and coord2.

        Parameters:
            coord1 (np.ndarray): The first set of coordinates.
            coord2 (np.ndarray): The second set of coordinates.
            direction (int): The direction to interpolate. 0 for longitudinal, 1 for latitudinal.
            speed (float): The speed to interpolate.

        Returns:
            list: A list of interpolated coordinates.
        """
        coord_df = pd.DataFrame({
            'coord_1_x': coord1[:, 0], 'coord_1_y': coord1[:, 1],
            'coord_2_x': coord2[:, 0], 'coord_2_y': coord2[:, 1],
        })
        coord_df['distance'] = self.haversine_distance(coord1, coord2)
        coord_df['minute'] = coord_df['distance'] // speed + 1
        coord_df['coord_speed'] = (coord_df[f'coord_2_{["x", "y"][direction]}'] - coord_df[
            f'coord_1_{["x", "y"][direction]}']) / coord_df['minute']  # direction = 0 横向，1 纵向
        coord_df['coord'] = coord_df.apply(self.generate_coords, args=(direction,), axis=1)
        return coord_df['coord'].tolist()

    def generate_coords(self, row, direction):
        """
        Generate a set of coordinates for a given row of data.

        Parameters:
            row (pd.Series): A row of data.
            direction (int): The direction to interpolate. 0 for longitudinal, 1 for latitudinal.

        Returns:
            list: A list of coordinates.
        """
        return [(row['coord_1_x'] + row['coord_speed'] * i if direction == 0 else row['coord_1_x'],
                 row['coord_1_y'] if direction == 0 else row['coord_1_y'] + row['coord_speed'] * i)
                for i in range(int(row['minute']))]

    def update_impedance(self, route_driver_num):  # 更新道路阻抗
        """
        Update the impedance of the road network based on the number of drivers on each route.

        Parameters:
            route_driver_num (np.ndarray or list): The number of drivers on each route.
        """
        self.current_edges['cars'] = route_driver_num
        self.current_edges['times'] = self.current_edges['free_time'] * \
                                      (1 + ALPHA * ((self.current_edges['cars'] / self.current_edges['capacity']) ** BETA))


class RoadChangedRoadNetwork(RoadNetwork):
    """
    This class extends the 'RoadNetwork' class to simulate changes in the road network. This could be used
    for scenarios such as road closures, accidents, or changes in road conditions. The class keeps track of
    changes in the road network over time and provides methods to update the current road network information.

    Parameters:
        city (str): The name of the city.
        drive_speed (float): The average driving speed.
        walk_speed (float): The average walking speed.
        where_change (list, optional): A list specifying the locations of the changes in the road network.
        when_change (list, optional): A list specifying the times of the changes in the road network.
        can_drive (list, optional): A list specifying whether it is possible to drive on the changed roads.
        can_walk (list, optional): A list specifying whether it is possible to walk on the changed roads.

    Attributes:
        changed_road_list, changed_edge_list, changed_node_list: They hold the changes in the roads, edges, and nodes
        respectively.
        current_road: Holds the current road network information.
        current_nodes: Holds the current nodes information.
        current_edges: Holds the current edges information.
        current_can_walk: Holds information about whether it is possible to walk on the current roads.
        current_can_drive: Holds information about whether it is possible to drive on the current roads.
        now_change_area: Holds the information about the current change area.
        now_change_time_range: Holds the information about the current change time range.
    """
    def __init__(
            self, city: str, drive_speed: float, walk_speed: float, where_change: list = None,
            when_change: list = None, can_drive: list = None, can_walk: list = None
    ):
        super().__init__(city, drive_speed, walk_speed)

        self.where_change = where_change
        self.when_change = when_change
        self.can_drive = can_drive
        self.can_walk = can_walk

        self.changed_road_list, self.changed_edge_list, self.changed_node_list = self.create_changed_road()

        self.current_road = None
        self.current_nodes = None
        self.current_edges = None
        self.current_can_walk = None
        self.current_can_drive = None

        self.now_change_area = None
        self.now_change_time_range = None
        self.update_current_road_info(env_time=0)

    def create_changed_road(self):
        """
        Create a list of updated road networks based on the locations and times of the changes.

        Returns:
            Tuple: A tuple containing lists of the changed road networks, edges, and nodes respectively.
        """
        road_list, edge_list, node_list = [], [], []
        for i in range(len(self.where_change)):
            # 需要更改的是道路编号
            where_change_i, when_change_i = self.where_change[i], self.when_change[i]
            origin_road_node, origin_road_edge = self.gdf_nodes, self.gdf_edges

            changed_road_edge = origin_road_edge[~origin_road_edge.index.isin(where_change_i)]
            changed_road_node = list(set(changed_road_edge['u'].tolist() + changed_road_edge['v'].tolist()))
            changed_road_node = origin_road_node[origin_road_node.index.isin(changed_road_node)]
            changed_road = pandana.Network(
                changed_road_node["x"], changed_road_node["y"], changed_road_edge["u"], changed_road_edge["v"],
                changed_road_edge[["length", "times"]], twoway=False,
            )
            road_list.append(changed_road)
            edge_list.append(changed_road_edge)
            node_list.append(changed_road_node)
        return road_list, edge_list, node_list

    def update_current_road_info(self, env_time):
        """
        Updates the current road, edges, and nodes based on the specified environment time. Also updates whether
        it's possible to walk or drive in the current network and the current change area and time range.

        Parameters:
            env_time (int): The current time in the environment.
        """
        road_index = self.find_interval(env_time)
        print(env_time, road_index)
        if road_index != -1:
            self.current_road = self.changed_road_list[road_index]
            self.current_edges = self.changed_edge_list[road_index]
            self.current_nodes = self.changed_node_list[road_index]

            self.current_can_walk = self.can_walk[road_index]
            self.current_can_drive = self.can_drive[road_index]

            self.now_change_area = self.where_change[road_index]
            self.now_change_time_range = np.arange(self.when_change[road_index][0], self.when_change[road_index][1])
        else:
            self.current_road = self.pandana_net
            self.current_edges = self.gdf_edges
            self.current_nodes = self.gdf_nodes
            self.current_can_walk = True
            self.current_can_drive = True

            self.now_change_area = []
            self.now_change_time_range = []

    def find_interval(self, env_time):
        """
        Finds the interval within which the given time falls in the change time list.

        Parameters:
            env_time (int): The current time in the environment.

        Returns:
            int: The index of the interval within which the given time falls. Returns -1 if the time is not within any
            of the intervals.
        """
        for i in range(len(self.when_change)):
            if self.when_change[i][0] <= env_time < self.when_change[i][1]:
                return i
        return -1

    def judge_nodes_in_control_area(self, nodes):
        """
        Judges whether the specified nodes are in the control area of the current road network.

        Parameters:
            nodes (list): A list of node ids.

        Returns:
            List: A list of boolean values indicating whether the corresponding node is in the control area.
        """
        changed = [True if node not in self.current_nodes.index else False for node in nodes]
        return changed

    def judge_routes_in_control_area(self, routes, env_time):
        """
        Judges whether the specified routes are in the control area at the specified time.

        Parameters:
            routes (list): A list of routes. Each route is a list of node ids.
            env_time (int): The current time in the environment.

        Returns:
            List: A list of boolean values indicating whether the corresponding route is in the control area.
        """
        changed = []
        for route in routes:
            route = np.array(route)
            route_time = np.arange(env_time, env_time + route.shape[0])
            route_changed = []
            for index in range(len(self.where_change)):
                now_where_changed = self.where_change[index]
                now_when_changed = self.when_change[index]
                in_control_area = np.in1d(route, now_where_changed)
                in_control_time = np.in1d(route_time, now_when_changed)
                control_link = np.any(in_control_area & in_control_time)
                route_changed.append(control_link)
            changed.append(np.any(route_changed))
        return changed

    def judge_od_have_path(self, dep_nodes, arr_nodes, dynamic_imp):
        """
        Judges whether there is a path between each pair of departure and arrival nodes in the current road network.

        Parameters:
            dep_nodes (list): A list of departure node ids.
            arr_nodes (list): A list of arrival node ids.

        Returns:
            List: A list of boolean values indicating whether there is a path for the corresponding pair of nodes.
            True, have path, False, do not have path.
        """
        if dynamic_imp:
            imp_name = 'times'
        else:
            imp_name = 'length'
        paths = self.current_road.shortest_paths(dep_nodes, arr_nodes, imp_name)
        paths_lengths = np.array(list(map(lambda x: x.shape[0], paths)))
        no_path = paths_lengths != 0
        return no_path

    def calculate_escape_route(self, coordinates, dynamic_impedance):
        """
        Calculates the escape routes for a set of coordinates based on the current road network and impedance.

        Parameters:
            coordinates (np.ndarray): A numpy array of coordinates.
            dynamic_impedance (bool): Indicates whether the impedance is dynamic.

        Returns:
            Tuple: A tuple of escape trajectories and routes.
        """
        origin_nodes = self.pandana_net.get_node_ids(coordinates[:, 0], coordinates[:, 1])
        target_nodes = self.current_road.get_node_ids(coordinates[:, 0], coordinates[:, 1])
        target_coordinates = self.get_node_coordinates(target_nodes)
        escape_trajectories, escape_routes = self.get_trajectories_from_origin_road(
            origin_nodes, target_nodes, coordinates, target_coordinates, dynamic_impedance
        )
        return escape_trajectories, escape_routes

    def get_trajectories_from_origin_road(self, dep_nodes, arr_nodes, dep_coords, arr_coords, dynamic_impedance):
        """
        Gets the trajectories from the departure nodes to the arrival nodes on the original road network.

        Args:
            dep_nodes (list): A list of departure node ids.
            arr_nodes (list): A list of arrival node ids.
            dep_coords (np.ndarray): A numpy array of departure coordinates.
            arr_coords (np.ndarray): A numpy array of arrival coordinates.
            dynamic_impedance (bool): Indicates whether the impedance is dynamic.

        Returns:
            Tuple: A tuple of trajectories and routes.
        """
        imp_name = 'times' if dynamic_impedance else 'length'
        dep_coords = dep_coords.tolist()
        arr_coords = arr_coords.tolist()
        routes = self.pandana_net.shortest_paths(dep_nodes, arr_nodes, imp_name=imp_name)
        link_origin = []
        end_origin = []
        no = []
        passenger = []
        all_all_trajectories = [[] for _ in routes]
        all_all_routes = [[] for _ in routes]
        not_at_same_node = []  # the indexes of passengers and drivers who are not on the same node
        for index, route in enumerate(routes):
            route = list(route)
            if len(route) == 1:
                all_all_trajectories[index] = [dep_coords[index]]
                all_all_routes[index] = [self.gdf_edges[self.gdf_edges['u'] == routes[index][0]].iloc[0]['index']]
            else:
                not_at_same_node.append(index)
                link_origin += route[:-1]
                end_origin += route[1:]
                no += list(range(len(route) - 1))
                passenger += [index] * (len(route) - 1)

        if len(not_at_same_node) != 0:
            links_shp = pd.merge(
                pd.DataFrame({
                    'u': link_origin,
                    'v': end_origin,
                    'key': [0] * len(link_origin),
                    'no': no, 'passenger': passenger
                }), self.gdf_edges, left_on=['u', 'v', 'key'], right_on=['u', 'v', 'key'], how='left'
            )

            if imp_name == 'length':
                links_shp['sum_length'] = links_shp.groupby('passenger')['length'].cumsum()
                links_shp['on_label'] = links_shp['sum_length'] // self.drive_speed
                links_shp_sum = gpd.GeoDataFrame(links_shp).dissolve(
                    by='passenger', aggfunc={'length': 'sum', 'passenger': 'mean'}
                )
                links_shp_sum['num_points'] = (links_shp_sum['length'] // self.drive_speed).astype(int)
            elif imp_name == 'times':
                links_shp['sum_times'] = links_shp.groupby('passenger')['times'].cumsum()
                links_shp['on_label'] = links_shp['sum_times'].astype(int)
                links_shp_sum = gpd.GeoDataFrame(links_shp).dissolve(
                    by='passenger', aggfunc={'times': 'sum', 'passenger': 'mean'}
                )
                links_shp_sum['num_points'] = links_shp_sum['times'].astype(int)
            else:
                raise NotImplementedError

            passenger_label_max = links_shp.groupby('passenger').agg({'on_label': 'max'})['on_label'].to_dict()
            road = pd.merge(
                links_shp[['index', 'on_label', 'passenger']].drop_duplicates(subset=['on_label', 'passenger'],
                                                                              keep='last'),
                pd.DataFrame([{'passenger': key, 'on_label': i} for key in passenger_label_max.keys()
                              for i in range(int(passenger_label_max[key] + 1))]),
                left_on=['passenger', 'on_label'], right_on=['passenger', 'on_label'], how='right'
            )

            road['index'] = road.groupby('passenger', group_keys=False)['index'].apply(lambda x: x.fillna(method='ffill'))
            road['index'] = road.groupby('passenger', group_keys=False)['index'].apply(lambda x: x.fillna(method='bfill'))
            road['index'] = road['index'].apply(int)
            all_road = [road[road['passenger'] == p]['index'].tolist() for p in list(road['passenger'].unique())]

            links_shp_sum['inter_position'] = links_shp_sum.apply(lambda x: [x['geometry'].interpolate(
                i / x['num_points'], normalized=True).coords[0] for i in range(1, x['num_points'] + 1)], axis=1)

            trajectories = links_shp_sum['inter_position'].values

            all_trajectories = [trajectory + [arr_coords[index]] for index, trajectory in enumerate(trajectories)]
            for index, od_pair in enumerate(not_at_same_node):
                all_all_trajectories[od_pair] = all_trajectories[index]
                all_all_routes[od_pair] = all_road[index]
        return all_all_trajectories, all_all_routes

    def update_impedance(self, route_driver_num):  # 更新道路阻抗
        """
        Update the impedance of the road network based on the number of drivers on each route.

        Parameters:
            route_driver_num (np.ndarray or list): The number of drivers on each route.
        """
        route_driver_num = pd.DataFrame({'index': self.current_edges.index.tolist(), 'cars': route_driver_num})
        route_driver_num.set_index('index', inplace=True)
        for road in self.changed_edge_list:
            road.update(route_driver_num)
            road.loc[:, 'times'] = road['free_time'] * (1 + ALPHA * ((road['cars'] / road['capacity']) ** BETA))
        self.gdf_edges.update(route_driver_num)
        self.gdf_edges.loc[:, 'times'] = self.gdf_edges['free_time'] * \
                                  (1 + ALPHA * ((self.gdf_edges['cars'] / self.gdf_edges['capacity']) ** BETA))



