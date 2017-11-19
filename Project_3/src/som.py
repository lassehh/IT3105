import matplotlib.pyplot as PLT
import numpy as np
import miscfunctions as misc
import time
import math


class SOM:
    trainingCases = None                   # City coordinates, scaled between 0 and 1
    testingCases = None
    trainingCaseLabels = None
    testingCaseLabels = None

    plotInterval = None
    testInterval = None

    weights = None                  # Numpy array of weights between input and output layer
    timeStep = None
    numOutputs = None
    gridSize = None

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



    def __init__(self, plotInterval = 10, testInterval = 10, problemType = 'TSP', problemArg = 1,
                 initialWeightRange = (0,1), gridSize = 10, epochs = 200, sigma_0 = 5.0,
                 tau_sigma = 1, eta_0 = 0.1, tau_eta = 1, fillIn = False, nmbrOfCases = None):
        self.sigma_0 = sigma_0
        self.tau_sigma = tau_sigma
        self.eta_0 = eta_0
        self.tau_eta = tau_eta

        self.initialWeightRange = initialWeightRange
        self.problemType = problemType
        if problemType == 'ICP':
            self.fillIn = fillIn
            # Plot and testing intervals
            self.plotInterval = plotInterval
            self.testInterval = testInterval
            # Generate training and testing cases
            case_generator = (lambda: misc.generate_mnist_data())
            caseFrac = (nmbrOfCases+600) / 55000
            testFrac = 600 / (600 + nmbrOfCases)
            self.caseManager = Caseman(cfunc=case_generator, cfrac=caseFrac, tfrac=testFrac)
            trainingCases = self.caseManager.get_training_cases()
            self.trainingCaseLabels = trainingCases[:, -1]
            self.trainingCases = trainingCases[:, :-1]
            testingCases = self.caseManager.get_testing_cases()
            self.testingCaseLabels = testingCases[:, -1]
            self.testingCases = testingCases[:, :-1]
            # Specify grid size and node labels
            self.numOutputs = gridSize*gridSize
            self.gridSize = gridSize
            self.nodeLabels = None
        elif problemType == 'TSP':
            self.plotInterval = plotInterval
            case_generator = (lambda: misc.generate_tsp_data(problemArg))
            self.caseManager = Caseman(cfunc=case_generator, cfrac=1.0, tfrac=0.0)
            self.trainingCases = self.caseManager.get_training_cases()
            self.unnormedTrainingCases = self.caseManager.get_unnormalized_cases()
            self.numOutputs = round(2 * len(self.trainingCases))  # The number of cities in the problem
        else:
            raise AssertionError("Unknown problem type " + problemType + ".")

        self.discriminantsStorage = [None] * self.numOutputs
        self.timeStep = 0
        self.epochs = epochs
        self.weight_initialization()


    def weight_initialization(self):
        if self.problemType == "TSP":
            self.weights = np.zeros(shape=(self.numOutputs, len(self.trainingCases[0])))
            index = 0
            circleRange = np.arange(0, 2 * math.pi, 2 * math.pi / self.numOutputs)
            for rad in circleRange:
                x = (math.cos(rad) + 1) / 2
                y = (math.sin(rad) + 1) / 2
                self.weights[index, :] = x, y
                index += 1
        elif self.problemType == "ICP":
            (lower_w, upper_w) = self.initialWeightRange
            self.weights = np.random.uniform(lower_w, upper_w, size=(self.numOutputs, len(self.trainingCases[0])))


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
            gridIndexNi = misc.index_list_2_grid(neuron_i, self.gridSize)
            gridIndexNj = misc.index_list_2_grid(neuron_j, self.gridSize)
            return (abs(gridIndexNi[0] - gridIndexNj[0]) + abs(gridIndexNi[1] - gridIndexNj[1]))

    def weight_update(self, sigma, eta, input, winner):
        if self.problemType == "TSP":
            T_ji = self.topological_neighbourhood_function(sigma, winner, winner)
            w_j = self.weights[winner, :]
            delta_w_j = eta * T_ji * (input - w_j)
            self.weights[winner, :] = w_j + delta_w_j

            step = 1.0
            index = int(winner + step) % self.numOutputs
            while(1):
                T_ji = self.topological_neighbourhood_function(sigma, winner, index)
                if abs(step) >= round(self.numOutputs*0.2):
                    break
                else:
                    w_j = self.weights[index, :]
                    delta_w_j = eta * T_ji * (input - w_j)
                    self.weights[index, :] = w_j + delta_w_j

                    step = (step + step/abs(step))*(-1)
                    index = int((index + step) % self.numOutputs)
        elif self.problemType == "ICP":
            notVisited = [winner]
            discovered = {winner}
            while(notVisited):
                node = notVisited.pop(0)
                nodeCoordinates = misc.index_list_2_grid(node, self.gridSize)
                neighbours = misc.find_2d_eight_way_neighbours(nodeCoordinates, self.gridSize)
                for neighbour in neighbours:
                    neighbourIndex = misc.index_grid_2_list(neighbour, self.gridSize)
                    if neighbourIndex not in discovered:
                        T_ji = self.topological_neighbourhood_function(sigma, winner, neighbourIndex)
                        if T_ji < 0.:
                            notVisited = []
                            break
                        else:
                            w_j = self.weights[neighbourIndex, :]
                            delta_w_j = eta * T_ji * (input - w_j)
                            self.weights[neighbourIndex, :] = w_j + delta_w_j

                        notVisited.append(neighbourIndex)
                        discovered.add(neighbourIndex)


    def run(self):
        if self.problemType == "TSP":
            self.run_tsp2()
        elif self.problemType == "ICP":
            self.run_icp2()
        else:
            raise AssertionError("Problem type \'", self.problemType, "\' is not defined.")


    # Description: Update timestep each epoch
    def run_tsp(self):
        fig, ax, background, weightPts, inputPts = misc.create_tsp_plot(self.weights, self.trainingCases)

        for timeStep in range (0, self.epochs + 1):
            self.timeStep = timeStep

            eta = self.learning_rate_function()
            sigma = self.neighbourhood_size_function()
            for i in self.trainingCases:
                winner = self.competitive_process(i)
                self.weight_update(eta = eta, sigma = sigma , input = i, winner = winner)

            # Plot every PLOT_INTVEVAL
            if timeStep % self.plotInterval == 0:
                misc.update_tsp_plot(fig, ax, background, self.weights, weightPts,
                                     self.learning_rate_function(), timeStep, self.epochs,
                                    self.neighbourhood_size_function())

        pathLength = self.calc_path_length()
        print("Final path length: ", pathLength)
        self.calc_node_path_length()
        wait = input("PRESS ENTER TO CLOSE PLOT")
        PLT.close(fig)


    # Description: Updates timestep for each case
    def run_tsp2(self):
        fig, ax, background, weightPts, inputPts = misc.create_tsp_plot(self.weights, self.trainingCases)
        timeStep = 0

        for epoch in range (0, self.epochs + 1):
            self.timeStep = epoch
            eta = self.learning_rate_function()
            sigma = self.neighbourhood_size_function()
            for case in self.trainingCases:
                # Training: do weight updates with the training cases
                winner = self.competitive_process(case)
                self.weight_update(eta=eta, sigma=sigma, input=case, winner=winner)

            #timeStep += 1
            # Plot every PLOT_INTVEVAL
            if epoch % self.plotInterval == 0:
                misc.update_tsp_plot(fig, ax, background, self.weights, weightPts,
                                     self.learning_rate_function(), epoch, self.epochs,
                                     self.neighbourhood_size_function())
            np.random.set_state(self.caseManager.state)
            np.random.shuffle(self.trainingCases)
            np.random.set_state(self.caseManager.state)
            np.random.shuffle(self.unnormedTrainingCases)

        pathLength = self.calc_path_length(plot = True)
        print("Final path length: ", pathLength)
        self.calc_node_path_length()
        wait = input("PRESS ENTER TO CLOSE PLOT")
        PLT.close(fig)

    def run_icp(self):
        runStartTime = time.clock()
        # Plot initial node classifications
        print("Plotting initial node labels..")
        self.nodeLabels = self.decide_nodes_classification(fillIn = False)
        misc.draw_image_classification_graph(nodeLabelsMatrix=self.nodeLabels, gridSize=int(self.gridSize))

        accuracyHistory = []
        for timeStep in range(0, self.epochs + 1):
            print(">> TIMESTEP: ", timeStep)
            self.timeStep = timeStep

            eta = self.learning_rate_function()
            sigma = self.neighbourhood_size_function()
            print('Learning rate: \t\t\t%5.4f' % (eta))
            print('Neighbourhood size: \t%5.4f' % (sigma))

            # Training: do weight updates with the training cases
            start = time.clock()
            for i in self.trainingCases:
                winner = self.competitive_process(i)
                self.weight_update(eta = eta, sigma = sigma, input = i, winner = winner)
            end = time.clock()
            print('Weight update time: \t%5.4f' % (end-start))

            # Plot every plot interval
            if (timeStep % self.plotInterval == 0 or timeStep == self.epochs) and timeStep != 0:
                self.nodeLabels = self.decide_nodes_classification(fillIn = True)
                misc.draw_image_classification_graph(nodeLabelsMatrix=self.nodeLabels, gridSize=int(self.gridSize))

            # Test accuracy of the classificator every test interval
            if (timeStep % self.testInterval == 0 or timeStep == self.epochs) and timeStep != 0:
                self.nodeLabels = self.decide_nodes_classification(fillIn = True)
                accuracy = self.test_icp_accuracy(self.trainingCases, self.trainingCaseLabels, self.nodeLabels, caseType = "Training")
                accuracyHistory.append((timeStep, accuracy)) #x,y plot


        # Final testing on the testing cases
        self.test_icp_accuracy(self.testingCases, self.testingCaseLabels, self.nodeLabels, caseType="Final testing")
        misc.plot_training_history(accuracyHistory, xtitle = "Timestep [ ]", ytitle = 'Accuracy [%]', title = "SOM ICP ACCURACY")

        runEndTime = time.clock()
        print("Run time : " , runEndTime - runStartTime)
        wait = input("ENTER TO QUIT")


    def run_icp2(self):
        runStartTime = time.clock()
        # Plot initial node classifications
        print("Plotting initial node labels..")
        self.nodeLabels = self.decide_nodes_classification(fillIn=False)
        misc.draw_image_classification_graph(nodeLabelsMatrix=self.nodeLabels, gridSize=int(self.gridSize))

        accuracyHistory = []
        for epoch in range(0, self.epochs):
            for case in self.trainingCases:
                eta = self.learning_rate_function()
                sigma = self.neighbourhood_size_function()

                if(self.timeStep % 10 == 0):
                    print(">> TIMESTEP: ", self.timeStep)
                    print('Learning rate: \t\t\t%5.4f' % (eta))
                    print('Neighbourhood size: \t%5.4f' % (sigma))

                # Training: do weight updates with the training cases
                winner = self.competitive_process(case)
                self.weight_update(eta=eta, sigma=sigma, input=case, winner=winner)

                # If it's time to plot or test accuracy, find each nodes classification
                if ((self.timeStep + 1) % self.plotInterval == 0) or ((self.timeStep + 1) % self.testInterval == 0):
                    self.nodeLabels = self.decide_nodes_classification(fillIn=self.fillIn)

                # Plot every plot interval
                if ((self.timeStep + 1) % self.plotInterval == 0 and self.timeStep > 0):
                    misc.draw_image_classification_graph(nodeLabelsMatrix=self.nodeLabels, gridSize=int(self.gridSize))

                # Test accuracy of the classificator every test interval
                if ((self.timeStep + 1) % self.testInterval == 0 and self.timeStep > 0):
                    accuracy = self.test_icp_accuracy(self.trainingCases, self.trainingCaseLabels, self.nodeLabels, caseType="Training")
                    accuracyHistory.append((self.timeStep, accuracy))  # x,y plot
                self.timeStep += 1

        # Final testing on the testing cases
        self.nodeLabels = self.decide_nodes_classification(fillIn=self.fillIn)
        misc.draw_image_classification_graph(nodeLabelsMatrix=self.nodeLabels, gridSize=int(self.gridSize))
        self.test_icp_accuracy(self.trainingCases, self.trainingCaseLabels, self.nodeLabels, caseType="\nFinal Training")
        self.test_icp_accuracy(self.testingCases, self.testingCaseLabels, self.nodeLabels, caseType="Final testing")
        misc.plot_training_history(accuracyHistory, xtitle="Timestep [ ]", ytitle='Accuracy [%]', title="SOM ICP ACCURACY")

        runEndTime = time.clock()
        print("Run time : ", runEndTime - runStartTime)
        time.sleep(0.2)
        wait = input("PRESS ENTER TO CLOSE PLOT")
        PLT.close()


    def test_icp_accuracy(self, cases, casesLabels, nodeLabels, caseType):
        correct, incorrect = 0.0, 0.0
        for index, case in enumerate(cases):
            caseLabel = int(casesLabels[index])
            winnerNeuronIndex = self.competitive_process(case)
            x, y = misc.index_list_2_grid(winnerNeuronIndex, self.gridSize)
            if nodeLabels[x,y] == caseLabel:
                correct += 1.0
            else: incorrect += 1.0
        accuracy = (correct/(incorrect+correct))*100.0
        print('%s accuracy: %.2f %%' % (caseType, accuracy))
        return accuracy


    def decide_nodes_classification(self, fillIn = True):
        winnerMatrix = np.zeros((self.gridSize, self.gridSize, 10))
        for index, input in enumerate(self.trainingCases):
            label = int(self.trainingCaseLabels[index])
            winnerIndex = self.competitive_process(input)
            x, y = misc.index_list_2_grid(winnerIndex, self.gridSize)
            winnerMatrix[x, y, label] += 1
        nodeLabels = np.zeros((self.gridSize, self.gridSize))
        nonLabeledNodes = []
        for x in range(0,self.gridSize):
            for y in range(0,self.gridSize):
                label = np.argmax(winnerMatrix[x,y,:])
                if winnerMatrix[x,y,label] == 0:
                    nodeLabels[x,y] = -1
                    nonLabeledNodes.append((x,y))
                else: nodeLabels[x,y] = label
        if fillIn == True:
            nodeLabels = self.fill_in_non_classified(nodeLabels, nonLabeledNodes)
        return nodeLabels

    def fill_in_non_classified(self, nodeLabels, nonLabeledNodes):
        np.random.shuffle(nonLabeledNodes)  # Randomly shuffle all cases
        for nonLabeledNode in nonLabeledNodes:
            labelsAccumulator = np.zeros(10)
            x, y = nonLabeledNode
            neighbours = misc.find_2d_eight_way_neighbours((x, y), self.gridSize)
            for neighbour in neighbours:
                neighbourX, neighbourY = neighbour
                neighbourLabel = nodeLabels[neighbourX, neighbourY]
                if neighbourLabel == -1:
                    continue
                else:
                    labelsAccumulator[int(neighbourLabel)] += 1
            nodeLabel = np.argmax(labelsAccumulator)
            if labelsAccumulator[nodeLabel] == 0:
                nodeLabels[x, y] = -1
            else:
                nodeLabels[x, y] = nodeLabel
        return nodeLabels


    def calc_path_length(self, plot = False):
        winners = np.ones(len(self.trainingCases), dtype = np.int32)*(-1)    # array to be filled with the winning neuron for each city
        for i, inp in enumerate(self.trainingCases):
            winning_neuron = self.competitive_process(inp)
            winners[i] = winning_neuron
        mapCityIndex2OutputIndex = np.stack((np.arange(len(self.trainingCases)), winners), axis = 1)
        mapCityIndex2OutputIndex = mapCityIndex2OutputIndex[np.argsort(mapCityIndex2OutputIndex[:, 1])] # sort the array based on ascending output neuron index

        distance = 0
        prevCity = 0
        firstCity = 0
        cityCoordinates = self.unnormedTrainingCases
        print("Visited cities:")
        if plot:
            PLT.ion()
            fig = PLT.figure()
            PLT.title("Final path")
            PLT.plot(cityCoordinates[:, 0], cityCoordinates[:, 1], 'ro')
        # go through the array:
        for j, cityAndOutput in enumerate(mapCityIndex2OutputIndex):
            city, _ = cityAndOutput
            if j == 0:
                prevCity = city
                firstCity = city
            else:
                distance += (np.linalg.norm(cityCoordinates[city, :] - cityCoordinates[prevCity, :]))

            if plot:
                xpts = np.append(cityCoordinates[prevCity, 0], cityCoordinates[city, 0])
                ypts = np.append(cityCoordinates[prevCity, 1], cityCoordinates[city, 1])
                PLT.plot(xpts, ypts, 'bx--')
                PLT.show()
                PLT.pause(0.001)

            prevCity = city

            print("\t", j+1, ": (x, y) = (", int(cityCoordinates[city, 0]), ", ", int(cityCoordinates[city, 1]), ")")
        distance += (np.linalg.norm(cityCoordinates[firstCity, :] - cityCoordinates[prevCity, :]))
        if plot:
            xpts = np.append(cityCoordinates[prevCity, 0], cityCoordinates[firstCity, 0])
            ypts = np.append(cityCoordinates[prevCity, 1], cityCoordinates[firstCity, 1])
            PLT.plot(xpts, ypts, 'bx--')
            PLT.show()
            PLT.pause(0.001)
            PLT.ioff()
            wait = input("PRESS ENTER TO CLOSE PLOT")
            PLT.close(fig)
        return distance

    def calc_node_path_length(self):
        cityCoordinates = self.caseManager.get_unnormalized_cases()
        max_x = np.max(cityCoordinates[:, 0])
        min_x = np.min(cityCoordinates[:, 0])
        max_y = np.max(cityCoordinates[:, 1])
        min_y = np.min(cityCoordinates[:, 1])
        # Scale weights between min and max
        self.weights[:, 0] = (max_x - min_x)*self.weights[:, 0] + min_x
        self.weights[:, 1] = (max_y - min_y) * self.weights[:, 1] + min_y

        prev_output = 0
        first_output = 0
        distance = 0
        for j in range(0, self.numOutputs):
            output = self.weights[j, :]
            if j == 0:
                prev_output = output
                first_output = output

            distance += (np.linalg.norm(output - prev_output))
            prev_output = output
        distance += (np.linalg.norm(first_output - prev_output))
        print("Distance between output nodes: ", distance)



class Caseman():
    def __init__(self, cfunc, cfrac = .8, tfrac = .1):
        self.casefunc = cfunc  # Function used to generate all data cases from a dataset
        self.case_fraction = cfrac  # What fraction of the total data cases to use
        self.test_fraction = tfrac  # What fraction of the data to use for final testing
        self.training_fraction = 1 - (tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        cases = self.cases[0]

        self.state = np.random.get_state()
        np.random.shuffle(cases)  # Randomly shuffle all cases

        if self.case_fraction < 1:
            case_separator = round(len(self.cases[0]) * self.case_fraction)
            cases = cases[0:case_separator]  # only use a fraction of the cases

        training_separator = round(len(cases) * self.training_fraction)
        self.training_cases = cases[0:training_separator]
        self.testing_cases = cases[training_separator:]

    def get_training_cases(self): return self.training_cases

    def get_testing_cases(self): return self.testing_cases

    def get_unnormalized_cases(self):
        cases = self.cases[1]
        np.random.set_state(self.state)
        np.random.shuffle(cases)
        return cases

# SOM 2nd BEST: 95% training, 75/75/75% testing
# grid 30, epochs 20, sigma0 16, tausigma 9 eta0 0.2

# SOM this also worked pretty well
# icpSOM = SOM(problemType = 'ICP', problemArg = 8, gridSize = 30, initialWeightRange = (0.49,0.51),
#             epochs = 5, sigma_0 = 16, tau_sigma = 1, eta_0 = 0.5, tau_eta = 6000,
#             plotInterval = 20, testInterval = 3)




# SOM BEST: 96% training, 76/82/80/80/78/79/74/82% testing
# icpSOM = SOM(problemType = 'ICP', problemArg = 8, gridSize = 40, initialWeightRange = (0.49,0.51),
#             epochs = 10, sigma_0 = 20, tau_sigma = 4, eta_0 = 0.2, tau_eta = 6000,
#             plotInterval = 10, testInterval = 2)

#with run_icp2 BESTBESTBESTBEST 17-11-2017:2220
#icpSOM = SOM(problemType = 'ICP', problemArg = None, gridSize = 40, initialWeightRange = (0,1),
            # epochs = 1, sigma_0 = 16, tau_sigma = 1400, eta_0 = 0.6, tau_eta = 2000,
            # plotInterval = 3500, testInterval = 3500, fillIn = True, nmbrOfCases = 3500)
# icpSOM = SOM(problemType = 'ICP', problemArg = None, gridSize = 10, initialWeightRange = (0s,1),
#             epochs = 50, sigma_0 = 5, tau_sigma = 50*250, eta_0 = 0.1, tau_eta = 50*30000,
#             plotInterval = 10000, testInterval = 10000, fillIn = True, nmbrOfCases = 600)
#icpSOM.run()

#tspSOM = SOM(problemType = 'TSP', problemArg = 4, plotInterval = 1000,
#               epochs = 350, sigma_0 = 5, tau_sigma = 15000, eta_0 = 0.9, tau_eta = 29000, fillIn = True)

#tspSOM.run()

#
#
#
#
#icpSOM = SOM(problemType = 'ICP', problemArg = None, gridSize = 40, initialWeightRange = (0,1),
            # epochs = 1, sigma_0 = 16, tau_sigma = 1400, eta_0 = 0.6, tau_eta = 2000,
            # plotInterval = 3500, testInterval = 3500, fillIn = True, nmbrOfCases = 3500)
#icpSOM.run()
