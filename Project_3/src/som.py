import matplotlib.pyplot as PLT
import numpy as np
import miscfunctions as misc
import time
import math

# Define constants
PLOT_INTERVAL = 5

class SOM:
    inputs = None
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



    def __init__(self, problemType = 'TSP', problemArg = 1, initialWeightRange = (0,1), num_outputs = 10, epochs = 200, sigma_0 = 5.0, tau_sigma = 1, eta_0 = 0.1, tau_eta = 1):
        self.sigma_0 = sigma_0
        self.tau_sigma = tau_sigma
        self.eta_0 = eta_0
        self.tau_eta = tau_eta

        self.initialWeightRange = initialWeightRange
        self.problemType = problemType
        if problemType == 'ICP':
            self.num_outputs = num_outputs
        elif problemType == 'TSP':
            case_generator = (lambda: misc.generate_tsp_data(problemArg))
            cMan = Caseman(cfunc=case_generator, cfrac=1.0, vfrac=0.0, tfrac=0.0)
            self.inputs = cMan.get_training_cases()
            self.num_outputs = round(2*len(self.inputs))  # The number of cities in the problem
        else:
            raise AssertionError("Unknown problem type " + problemType + ".")

        self.timeStep = 0
        self.epochs = epochs




    def weight_initialization(self):
        (lower_w, upper_w) = self.initialWeightRange
        self.weights = np.zeros(shape=(self.num_outputs, len(self.inputs[0])))
        index = 0
        range = np.arange(0, 2*math.pi, 2*math.pi/self.num_outputs)
        for rad in range:
            x = (math.cos(rad) + 1) / 2
            y = (math.sin(rad) + 1) / 2
            self.weights[index, :] = x, y
            index += 1

        # each column represents the weights entering one output neuron
        #self.weights = np.random.uniform(lower_w, upper_w, size=(self.num_outputs, len(self.inputs[0])))

    def normalize_weights(self):
        for j in range(0, self.num_outputs):
            w_j = self.weights[j, :]
            self.weights[j, :] = misc.normalize(w_j)


    # the discriminant is the squared Euclidean distance between the input vector and the weight vector w_j for each neuron j.
    def discriminant_function(self, input, w_j):
        d_j = (np.linalg.norm(input - w_j))**2
        return d_j

    # for each input pattern, each neuron j computes the value of the discriminant function
    # the neuron with the smallest discriminant wins
    def competitive_process(self, input):
        discriminants = []
        for j in range(0, self.num_outputs):
            w_j = self.weights[j, :]
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
        sigma = (self.sigma_0)*np.exp(-self.timeStep/self.tau_sigma)
        if sigma < 0.4:
            sigma = 0.4
        return sigma

    def learning_rate_function(self):
        eta = self.eta_0*np.exp(-self.timeStep/self.tau_eta)
        return eta

    def manhattan_distance(self, neuron_i, neuron_j):
        if self.problemType == 'TSP':
            # The output is shaped like a ring
            distance = abs(neuron_i - neuron_j)
            if distance >= self.num_outputs/2:
                # the calculated distance is not the shortest possible distance
                distance = abs(self.num_outputs - distance)
            return distance
        elif self.problemType == 'ICP':
            pass

    def weight_update(self, input, winner):
        eta = self.learning_rate_function()
        weights_updated = np.zeros((self.num_outputs, len(self.inputs[0])))

        T_ji = self.topological_neighbourhood_function(winner, winner)
        w_j = self.weights[winner, :]
        delta_w_j = eta * T_ji * (input - w_j)
        self.weights[winner, :] = w_j + delta_w_j

        lowTopFunc = 0
        step = 1.0
        index = int(winner + step) % self.num_outputs
        while lowTopFunc == 0:
            T_ji = self.topological_neighbourhood_function(winner, index)
            if T_ji < 0.00001:
                lowTopFunc = 1
            else:
                w_j = self.weights[index, :]
                delta_w_j = eta * T_ji * (input - w_j)
                self.weights[index, :] = w_j + delta_w_j

                step = (step + step/abs(step))*(-1)
                index = int((index + step) % self.num_outputs)


    def tsp_plot(self, fig, doblit = True):
        pass
        # x, y = rw.next()
        # points.set_data(x, y)
        #
        # if doblit:
        #     # restore background
        #     fig.canvas.restore_region(background)
        #
        #     # redraw just the points
        #     ax.draw_artist(points)
        #
        #     # fill in the axes rectangle
        #     fig.canvas.blit(ax.bbox)
        #
        # else:
        #     # redraw everything
        #     fig.canvas.draw()

        # plt.clf()
        # plt.title("Epoch: " + str(self.timeStep) + "/" + str(self.epochs) + ". Learning rate: " + str(
        #     self.learning_rate_function()) +
        #           ". Neighbourhood: " + str(self.neighbourhood_size_function()))
        # neuronRingY = np.append(self.weights[:, 0], self.weights[0, 0])
        # neuronRingX = np.append(self.weights[:, 1], self.weights[0, 1])
        # ax = fig.gca()
        # ax.set_xticks(np.arange(0, 1, 0.1))
        # ax.set_yticks(np.arange(0, 1, 0.1))
        # plt.plot(neuronRingY, neuronRingX, 'bx--')
        # plt.pause(0.0001)
        # plt.plot(self.inputs[:, 0], self.inputs[:, 1], 'g^')
        # plt.pause(0.0001)
        # plt.grid(True)
        # plt.pause(0.0001)
        # plt.show()

    def run(self):
        fig, ax = PLT.subplots(1, 1)
        ax.set_aspect('equal')

        major_ticks = np.arange(-0.1, 1.1, 0.1)
        minor_ticks = np.arange(-0.1, 1.1, 0.02)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        background = fig.canvas.copy_from_bbox(ax.bbox)

        self.weight_initialization()
        neuronRingY = np.append(self.weights[:, 0], self.weights[0, 0])
        neuronRingX = np.append(self.weights[:, 1], self.weights[0, 1])

        neurons = ax.plot(neuronRingY, neuronRingX, 'bx--')[0]
        inputs = ax.plot(self.inputs[:, 0], self.inputs[:, 1], 'g^')

        for timeStep in range (1, self.epochs + 1):
            self.timeStep = timeStep

            startTime = time.clock()
            for i in self.inputs:
                winner = self.competitive_process(i)
                self.weight_update(i, winner)
            endTime = time.clock()
            print("Weight update time: \t", endTime - startTime, "\t[s]")

            if timeStep % PLOT_INTERVAL == 0:
                startTime = time.clock()
                neuronRingY = np.append(self.weights[:, 0], self.weights[0, 0])
                neuronRingX = np.append(self.weights[:, 1], self.weights[0, 1])
                neurons.set_data(neuronRingY, neuronRingX)

                PLT.pause(0.00001)

                fig.suptitle("Epoch: " + str(self.timeStep) + "/" + str(self.epochs) + ". Learning rate: " + str(
                    self.learning_rate_function()) +
                             ". Neighbourhood: " + str(self.neighbourhood_size_function()), fontsize=12)
                fig.canvas.restore_region(background)
                ax.draw_artist(neurons)
                fig.canvas.blit(ax.bbox)

                endTime = time.clock()
                print("Plot time: \t\t\t\t", endTime - startTime, "\t[s]")

        wait = input("ENTER TO QUIT")
        PLT.close(fig)
        PLT.pause(0.01)


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

testSOM = SOM(problemType = 'TSP', problemArg = 1, initialWeightRange = (0,1),
              epochs = 300, sigma_0 = 5.0, tau_sigma = 100, eta_0 = 0.3, tau_eta = 2000)
testSOM.run()

