from static.Simulation import Simulation
from static.SimulationReplay import SimulationReplay


def begin_simulation(**kwargs):
    simulation = Simulation(**kwargs)

    simulation.simulate()

    simulation.summary()

    # simulation.results_analysis()
    return simulation


def replay_simulation():
    pass


if __name__ == '__main__':
    name = 'Test1'
    city = 'beijing'  # todo: city可以为经纬度范围，也可以为str。如果是str，就判断是否存储有该城市的路网，如果是经纬度范围，则直接下载路网
    platform_path = r'./static/platform.yaml'
    time_limit = 1440   # 仿真时长，默认每个时间窗时长为1min
    drive_speed = 420  # 静态阻抗下，车辆的行驶速度 m/min
    walk_speed = 100  # 行人的步行速度 m/min
    road_change = True  # 是否发生了道路状况改变？
    if road_change:
        where_change = []  # 列表，存储道路管控的空间范围坐标
        when_change = []  # 列表，存储道路管控的时间范围
        can_drive = [False]
        can_walk = [False]
