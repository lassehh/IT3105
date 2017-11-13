import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import copy
import random
import time
from tensorflow.examples.tutorials.mnist import input_data

labelColorDict = {0: 'b', 1: 'g', 2: 'y', 3: 'r', 4: 'pink', 5: 'm', 6:
                  'cyan', 7: 'grey', 8: 'black', 9: 'orange'}



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

def create_tsp_plot(weights, inputs):
    fig, ax = plt.subplots(1, 1)
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

    neuronRingY = np.append(weights[:, 0], weights[0, 0])
    neuronRingX = np.append(weights[:, 1], weights[0, 1])


    inputPts = ax.plot(inputs[:, 0], inputs[:, 1], 'g^')
    weightPts = ax.plot(neuronRingY, neuronRingX, 'bx--')[0]
    plt.pause(0.00001)

    return fig, ax, background, weightPts, inputPts

def update_tsp_plot(fig, ax, background, weights, weightPts,
                    learningRate, timeStep, epochs, neighbourhood):
    neuronRingY = np.append(weights[:, 0], weights[0, 0])
    neuronRingX = np.append(weights[:, 1], weights[0, 1])
    weightPts.set_data(neuronRingY, neuronRingX)

    fig.suptitle("Epoch: " + str(timeStep) + "/" + str(epochs) + ". Learning rate: " + str(
        learningRate) + ". Neighbourhood: " + str(neighbourhood), fontsize=12)
    ax.draw_artist(weightPts)

    plt.pause(0.00001)

def draw_image_classification_graph(gridSize = 10, numberOfLabels = 9):
    start = time.clock()
    G = nx.grid_2d_graph(gridSize, gridSize)
    pos = dict((n, n) for n in G.nodes())

    for node in G:
        labelNumber = round(random.random()*numberOfLabels)
        labelColor = labelColorDict[labelNumber]
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[node],
                               node_color=labelColor,
                               node_size=600,
                               alpha=0.8)

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    end = time.clock()
    print("Graph plotting time : ", end-start)
    plt.axis('off')
    plt.show()


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm

def generate_mnist_data():
    data_dir = '../data/mnist'
    mnist = input_data.read_data_sets(data_dir, one_hot=False)
    trainingSet = mnist.train
    labels = np.array([trainingSet.labels])
    concatenated = np.concatenate((trainingSet.images, labels.T), axis = 1)
    return concatenated, concatenated

draw_image_classification_graph()
# generate_mnist_data()
