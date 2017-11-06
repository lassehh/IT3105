import matplotlib.pyplot as PLT
import numpy as np
import miscfunctions as misc
import time

class SOM:
    input = None
    weights = None                  # Numpy array of weights between input and output layer
    timeStep = None
    num_outputs = None

    caseFunction = None
    initialWeightRange = None

    # Time constants and, initial neighbour and learning rate values
    sigma_0 = None
    tau_sigma = None
    eta_0 = None
    tau_eta = None

    # TSP or image classification problem (ICP)
    problemType = None



    def __init__(self, cfunc, problemType = 'TSP', initialWeightRange = (-.1,.1), num_outputs = 10):

        self.initialWeightRange = initialWeightRange
        self.caseFunction = cfunc
        self.generate_cases()
        self.problemType = problemType
        if problemType == 'ICP':
            self.num_outputs = num_outputs
        elif problemType == 'TSP':
            self.num_outputs = len(self.input)      # The number of cities in the problem
        else:
            raise AssertionError("Unknown problem type " + problemType + ".")

        self.timeStep = 0


    def generate_cases(self):
        self.input = self.caseFunction()    # Run the case generator.  Case = input-vector

    def weight_initialization(self):
        (lower_w, upper_w) = self.initialWeightRange
        # each column represents the weights entering one output neuron
        self.weights = np.random.uniform(lower_w, upper_w, size=(len(self.input[0]), self.num_outputs))


    # the discriminant is the squared Euclidean distance between the input vector and the weight vector w_j for each neuron j.
    def discriminant_function(self, input, w_j):
        d_j = (np.linalg.norm(input - w_j))**2
        return d_j

    # for each input pattern, each neuron j computes the value of the discriminant function
    # the neuron with the smallest discriminant wins
    def competitive_process(self, input):
        discriminants = []
        for j in range(0, self.num_outputs):
            w_j = self.weights[:,j]
            d_j = self.discriminant_function(input, w_j)
            discriminants.append(d_j)
        winner = np.argmin(np.array(discriminants))
        return winner

    def topological_neighbourhood_function(self, winner, neuron_j):
        S_ji = self.manhattan_distance(winner, neuron_j)
        sigma = self.neighbourhood_size_function()
        T_ji = np.exp(-(S_ji**2)/(2*(sigma**2)))
        return T_ji

    def neighbourhood_size_function(self):
        sigma = self.sigma_0*np.exp(-self.timeStep/self.tau_sigma)
        return sigma

    def learning_rate_function(self):
        eta = self.eta_0*np.exp(-self.timeStep/self.tau_eta)
        return eta

    def manhattan_distance(self, neuron_i, neuron_j):
        if self.problemType == 'TSP':
            # The output is shaped like a ring
            distance = abs(neuron_i - neuron_j)
            if distance >= len(self.input)/2:
                # the calculated distance is not the shortest possible distance
                distance = len(self.input) - distance
            return distance
        elif self.problemType == 'ICP':
            pass


    def weight_update(self):
        eta = self.learning_rate_function()
        x = self.input[0]
        winner = self.competitive_process(x)
        weights_updated = np.zeros(len(self.input[0]), self.num_outputs)
        for j in range(0, self.num_outputs):
            T_ji = self.topological_neighbourhood_function(winner, j)
            w_j = self.weights[:,j]
            delta_w_j = eta*T_ji*(x - w_j)
            weights_updated[:,j] = w_j + delta_w_j
        self.weights = weights_updated



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


