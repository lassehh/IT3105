import matplotlib.pyplot as PLT
import numpy as np
import miscfunctions as misc
import time
import math

# Define constants
PLOT_INTERVAL = 5

class SOM:
    inputs = None                   # City coordinates, scaled between 0 and 1
    weights = None                  # Numpy array of weights between input and output layer
    timeStep = None
    numOutputs = None

    initialWeightRange = None
    caseManager = None

    # Time constants and, initial neighbour and learning rate values
    sigma_0 = None
    tau_sigma = None
    eta_0 = None
    tau_eta = None

    # TSP or image classification problem (ICP)
    problemType = None

    discriminantsStorage = None



    def __init__(self, problemType = 'TSP', problemArg = 1, initialWeightRange = (0,1), num_outputs = 10, epochs = 200, sigma_0 = 5.0, tau_sigma = 1, eta_0 = 0.1, tau_eta = 1):
        self.sigma_0 = sigma_0
        self.tau_sigma = tau_sigma
        self.eta_0 = eta_0
        self.tau_eta = tau_eta

        self.initialWeightRange = initialWeightRange
        self.problemType = problemType
        if problemType == 'ICP':
            case_generator = (lambda: misc.generate_mnist_data())
            self.caseManager = Caseman(cfunc=case_generator, cfrac=0.05, vfrac=0.0, tfrac=0.05)
            self.inputs = self.caseManager.get_training_cases()
            self.numOutputs = num_outputs
        elif problemType == 'TSP':
            case_generator = (lambda: misc.generate_tsp_data(problemArg))
            self.caseManager = Caseman(cfunc=case_generator, cfrac=1.0, vfrac=0.0, tfrac=0.0)
            self.inputs = self.caseManager.get_training_cases()
            self.numOutputs = round(2 * len(self.inputs))  # The number of cities in the problem
        else:
            raise AssertionError("Unknown problem type " + problemType + ".")

        self.discriminantsStorage = [None] * self.num_outputs
        self.timeStep = 0
        self.epochs = epochs




    def weight_initialization(self):
        (lower_w, upper_w) = self.initialWeightRange
        self.weights = np.zeros(shape=(self.numOutputs, len(self.inputs[0])))
        index = 0
        range = np.arange(0, 2 * math.pi, 2 * math.pi / self.numOutputs)
        for rad in range:
            x = (math.cos(rad) + 1) / 2
            y = (math.sin(rad) + 1) / 2
            self.weights[index, :] = x, y
            index += 1

        # each column represents the weights entering one output neuron
        #self.weights = np.random.uniform(lower_w, upper_w, size=(self.num_outputs, len(self.inputs[0])))


    # the discriminant is the squared Euclidean distance between the input vector and the weight vector w_j for each neuron j.
    def discriminant_function(self, input, w_j):
        d_j = (np.linalg.norm(input - w_j))**2
        return d_j

    # for each input pattern, each neuron j computes the value of the discriminant function
    # the neuron with the smallest discriminant wins
    def competitive_process(self, input):
        for j in range(0, self.numOutputs):
            w_j = self.weights[j, :]
            d_j = self.discriminant_function(input, w_j)
            self.discriminantsStorage[j] = d_j
        winner = np.argmin(np.array(self.discriminantsStorage))
        return winner

    def topological_neighbourhood_function(self, sigma, winner, neuron_j):
        S_ji = self.manhattan_distance(winner, neuron_j)
        T_ji = np.exp(-(S_ji**2)/(2*(sigma**2)))
        return T_ji

    def neighbourhood_size_function(self):
        sigma = (self.sigma_0)*np.exp(-self.timeStep/self.tau_sigma)
        return sigma

    def learning_rate_function(self):
        eta = self.eta_0*np.exp(-self.timeStep/self.tau_eta)
        return eta

    def manhattan_distance(self, neuron_i, neuron_j):
        if self.problemType == 'TSP':
            # The output is shaped like a ring
            distance = abs(neuron_i - neuron_j)
            if distance >= self.numOutputs/2:
                # the calculated distance is not the shortest possible distance
                distance = abs(self.numOutputs - distance)
            return distance
        elif self.problemType == 'ICP':
            pass

    def weight_update(self, sigma, eta, input, winner):
        T_ji = self.topological_neighbourhood_function(sigma, winner, winner)
        w_j = self.weights[winner, :]
        delta_w_j = eta * T_ji * (input - w_j)
        self.weights[winner, :] = w_j + delta_w_j

        lowTopFunc = 0
        step = 1.0
        index = int(winner + step) % self.numOutputs
        while lowTopFunc == 0:
            T_ji = self.topological_neighbourhood_function(sigma, winner, index)
            if T_ji < 0.001:
                lowTopFunc = 1
            else:
                if self.problemType == "TCP":
                    w_j = self.weights[index, :]
                    delta_w_j = eta * T_ji * (input - w_j)
                    self.weights[index, :] = w_j + delta_w_j

                    step = (step + step/abs(step))*(-1)
                    index = int((index + step) % self.num_outputs)
                elif self.problemType == "ICP":
                    pass

    def run(self):
        self.weight_initialization()

        fig, ax, background, weightPts, inputPts = misc.create_tsp_plot(self.weights, self.inputs)

        for timeStep in range (0, self.epochs + 1):
            self.timeStep = timeStep
            startTime = time.clock()
            eta = self.learning_rate_function()
            sigma = self.neighbourhood_size_function()
            for i in self.inputs:
                winner = self.competitive_process(i)
                self.weight_update(eta = eta, sigma = sigma , input = i, winner = winner)
            endTime = time.clock()
            print("Weight update time: \t", endTime - startTime, "\t[s]")

            if timeStep % PLOT_INTERVAL == 0:
                startTime = time.clock()
                misc.update_tsp_plot(fig, ax, background, self.weights, weightPts,
                                     self.learning_rate_function(), timeStep, self.epochs,
                                    self.neighbourhood_size_function())
                endTime = time.clock()
                print("Plot time: \t\t\t\t", endTime - startTime, "\t[s]")

        #path_length = self.calculate_path_length()
        wait = input("ENTER TO QUIT")
        PLT.ioff()
        #PLT.close(fig)
        PLT.pause(0.01)


    def calculate_path_length(self):
        distance = 0
        prevCity = 0
        firstCity = 0
        visitedCities = set()
        cityCoordinates = self.caseManager.unnormalized_cases

        # fig = PLT.figure()
        for j in range(0, self.numOutputs):
            discriminants = []
            w_j = self.weights[j, :]
            for input in self.inputs:
                d_j = self.discriminant_function(input, w_j)
                discriminants.append(d_j)
            currentCity = np.argmin(np.array(discriminants))

            # The current output neuron "won" input city with index "currentCity"
            # Calculate distance between this city and the previous city in the path
            if currentCity not in visitedCities or j == self.numOutputs-1:
                visitedCities.add(currentCity)
                if j == 0:
                    distance = 0
                    prevCity = currentCity
                    firstCity = currentCity
                elif j == self.numOutputs - 1:
                    # Last output neuron reached. Must calculate the distance from the last city to the first
                    if firstCity == currentCity:  # the last output neuron lies between the first and last city
                        distance += (np.linalg.norm(cityCoordinates[prevCity, :] - cityCoordinates[firstCity, :]))
                    else:
                        distance += (np.linalg.norm(cityCoordinates[currentCity, :] - cityCoordinates[firstCity, :]))
                else:
                    distance += (np.linalg.norm(cityCoordinates[currentCity, :] - cityCoordinates[prevCity, :]))
                    prevCity = currentCity

                print("Current path length: ", distance)
                # PLT.title("Current city")
                # PLT.plot(cityCoordinates[:, 0], cityCoordinates[:, 1], 'ro')
                # PLT.plot(cityCoordinates[currentCity, 0], cityCoordinates[currentCity, 1], 'bx')
                #
                # PLT.show()
                # PLT.pause(0.001)

        return distance

class Caseman():
    def __init__(self, cfunc, cfrac = .8, vfrac = 0.1, tfrac = .1):
        self.casefunc = cfunc  # Function used to generate all data cases from a dataset
        self.case_fraction = cfrac  # What fraction of the total data cases to use
        # self.validation_fraction = vfrac  # What fraction of the data to use for validation
        self.test_fraction = tfrac  # What fraction of the data to use for final testing
        self.training_fraction = cfrac # 1 - (tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        cases = self.cases[0]
        self.unnormalized_cases = self.cases[1]

        state = np.random.get_state()
        np.random.shuffle(cases)  # Randomly shuffle all cases
        np.random.set_state(state)
        np.random.shuffle(self.unnormalized_cases)

        if self.case_fraction < 1:
            case_separator = round(len(self.cases) * self.case_fraction)
            cases = cases[0:case_separator]  # only use a fraction of the cases

        training_separator = round(len(cases) * self.training_fraction)
        self.training_cases = cases[0:training_separator]
        self.testing_cases = cases[training_separator:]

    def get_training_cases(self): return self.training_cases

    def get_testing_cases(self): return self.testing_cases


testSOM = SOM(problemType = 'TSP', problemArg = 2, initialWeightRange = (0,1),
              epochs = 400, sigma_0 = 5.0, tau_sigma = 100, eta_0 = 0.3, tau_eta = 2000)
testSOM.run()

