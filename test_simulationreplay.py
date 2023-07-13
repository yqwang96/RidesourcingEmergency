from static.SimulationReplay import SimulationReplay


sp = SimulationReplay(r'.\Results\Test2\simulate_config.pickle')
sp.load_logger()

sp.results_analysis()
sp.visualization()

