from static.Simulation import Simulation


TIME_LIMIT = 1440
simulation = Simulation(
    name='Test2',
    city='beijing',
    platform_path=r'./static/platform.yaml',
    time_limit=TIME_LIMIT,
    drive_speed=420,
    walk_speed=100,
    road_changed=True,
    where_change=[range(600), range(100, 3000), range(100, 2000)],
    when_change=[[0, 200], [300, 500], [600, 1000]],
    can_drive=[False, False, False],
    can_walk=[False, True, False]
)

simulation.simulate()

simulation.summary()

simulation.results_analysis()

