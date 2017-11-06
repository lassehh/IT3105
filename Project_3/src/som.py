import matplotlib.pyplot as PLT
import numpy as np
import miscfunctions as misc
import time

class SOM:
    input = None
    weights = None                  # Numpy array of weights between input and output layer
    timeStep = None

    # Time constants and, initial neighbour and learning rate values
    sigma_0 = None
    tau_sigma = None
    eta_0 = None
    tau_eta = None

    def __init__(self):
        pass

    def weight_initialization(self):
        pass

    def discriminant_function(self):
        pass

    def topological_neighbourhood_function(self):
        pass

    def neightbourhood_function(self):
        pass

    def learning_rate_function(self):
        pass

    def manhattan_distance(self):
        pass

    def weight_update(self):
        pass

class Caseman():
    def __init__(self, cfunc, cfrac = .8, vfrac = .1, tfrac = .1):
        self.casefunc = cfunc  # Function used to generate all data cases from a dataset
        self.case_fraction = cfrac  # What fraction of the total data cases to use
        self.validation_fraction = vfrac  # What fraction of the data to use for validation
        self.test_fraction = tfrac  # What fraction of the data to use for final testing
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        cases = self.cases
        np.random.shuffle(cases)  # Randomly shuffle all cases
        if self.case_fraction < 1:
            case_separator = round(len(self.cases) * self.case_fraction)
            cases = cases[0:case_separator]  # only use a fraction of the cases

        training_separator = round(len(cases) * self.training_fraction)
        validation_separator = training_separator + round(len(cases) * self.validation_fraction)
        self.training_cases = cases[0:training_separator]
        self.validation_cases = cases[training_separator:validation_separator]
        self.testing_cases = cases[validation_separator:]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases


# print("Trying some dumb shit: ")
# dummySOM = SOM()
# timeSteps = 10000
# dummy_points_x = [1,2,3]
# dummy_points_y = [1,2,3]
#
# fig = PLT.figure()
# PLT.axis([0,10,0,1])
# PLT.ion()
# for step in range(0, timeSteps):
#     dummy_points_x[0] += 0.1
#     dummy_points_y[0] += 0.1
#     dummy_points_x[1] += 0.1
#     dummy_points_y[1] -= 0.1
#     PLT.clf()
#     PLT.plot(dummy_points_x, dummy_points_y)
#     PLT.show()
#     PLT.pause(0.0001)
#
#     time.sleep(0.1)


