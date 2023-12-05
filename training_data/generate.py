from utils.Generator import Generator

data_generator = Generator(result_path='data/test', runsim=True, maps_per_task=2,allDAG=True)

# data_generator = Generator(runsim=True, allDAG=False)
# max_out, alpha, beta = data_generator.getRandomDAGParameters(True)
# data_generator.generate(max_out, alpha, beta)



