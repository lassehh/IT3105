import numpy as np
import copy
from tensorflow.examples.tutorials.mnist import input_data




def generate_tsp_data(problemNumbr):
    with open('../data/TSP_Problems_Euclidean/' + str(problemNumbr) + '.txt', 'r') as f:
        firstLine = f.readline().strip("\n")
        firstLine = firstLine.split(":")
        name = firstLine[1].strip()
        secondLine = f.readline().strip("\n")
        secondLine = secondLine.split(":")
        problemType = secondLine[1].strip()
        thirdLine = f.readline().strip("\n")
        thirdLine = thirdLine.split(":")
        nmbrOfCities = thirdLine[1].strip()
        cities = np.zeros((int(nmbrOfCities), 2))
        cities_true_coordinates = np.zeros((int(nmbrOfCities), 2))
        # Skip two lines
        f.__next__()
        f.__next__()
        for line in f:
            line = line.strip("\n")
            if(line == "EOF"):
                break
            line = line.split()
            cityNmbr, x, y = line
            cityNmbr, x, y = int(cityNmbr), float(x), float(y)
            cities[cityNmbr-1,:] = x, y
            cities_true_coordinates[cityNmbr-1,:] = x, y
        max_x = np.max(cities[:, 0])
        min_x = np.min(cities[:, 0])
        max_y = np.max(cities[:, 1])
        min_y = np.min(cities[:, 1])
        cities[:, 0] = (cities[:, 0] - min_x) / (max_x - min_x)
        cities[:, 1] = (cities[:, 1] - min_y) / (max_y - min_y)
    return cities, cities_true_coordinates

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm

def generate_mnist_data():
    data_dir = '../data/mnist'
    mnist = input_data.read_data_sets(data_dir, one_hot=False)
    trainingSet = mnist.train
    lol = 0
    transpose = trainingSet.labels.T
    # a = np.concatenate((trainingSet.images, (trainingSet.labels).T), axis = 1)
    return trainingSet


# generate_mnist_data()
