from utils.Generator import Generator

# data_generator = Generator(result_path='data/test', run_sim=True, maps_per_task=2,all_dag=True, save_results=True)

data_generator = Generator(runsim=True, all_dag=False)
max_out, alpha, beta = data_generator.getRandomDAGParameters(True)
data_generator.generate(max_out, alpha, beta)



