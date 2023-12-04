from utils.Generator import Generator

data_generator = Generator(runsim=True)

max_out, alpha, beta = data_generator.getRandomDAGParameters(True)
data_generator.generate(max_out, alpha, beta)

