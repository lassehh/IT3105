import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import copy
import random
import time
from tensorflow.examples.tutorials.mnist import input_data

labelColorDict = {0: 'b', 1: 'g', 2: 'y', 3: 'r', 4: 'pink', 5: 'm', 6:
                  'cyan', 7: 'grey', 8: 'black', 9: 'orange', -1: 'white'}



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


    inputPts = ax.plot(inputs[:, 0], inputs[:, 1], '^', color='darkslateblue')
    weightPts = ax.plot(neuronRingY, neuronRingX, '.--', color='teal', linewidth=0.8)[0]
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



def draw_image_classification_graph(nodeLabelsMatrix, gridSize, numberOfLabels = 9):
    G = nx.grid_2d_graph(gridSize, gridSize)
    pos = dict((n, n) for n in G.nodes())
    for n in G:
        x, y = n
        if (x - 1 >= 0 and y - 1 >= 0):  # SW
            G.add_edge(n, (x - 1, y - 1))
        if (x + 1 < gridSize and y - 1 >= 0):  # SE
            G.add_edge(n, (x + 1, y - 1))
    labels = {}
    for node in G:
        labelNumber = nodeLabelsMatrix[node]
        labelColor = labelColorDict[labelNumber]
        labels[node] = int(labelNumber)
        nx.draw_networkx_nodes(G, pos, nodelist = [node], node_color = labelColor, node_size = 200, alpha = 0.65)
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    plt.axis('off')
    plt.show()

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm

def index_grid_2_list(coordinates, gridSize):
    return coordinates[0] + coordinates[1] * gridSize

def index_list_2_grid(index, gridSize):
    x, y = 0, 0
    while((y+1)*gridSize <= index and (y+1) < gridSize):
        y += 1
    while(x + y*gridSize != index):
        x += 1
        if(x >= gridSize): return AttributeError("Cannot convert index to x,y coordinates.")
    return (x,y)

def find_2d_four_way_neighbours(coordinates, gridSize):
    x, y = coordinates
    neighbours = []
    if(x + 1 < gridSize):
        neighbours.append((x + 1, y))
    if(x - 1 >= 0):
        neighbours.append((x - 1, y))
    if (y + 1 < gridSize):
        neighbours.append((x, y + 1))
    if (y - 1 >= 0):
        neighbours.append((x, y - 1))
    return neighbours

def find_2d_eight_way_neighbours(coordinates, gridSize):
    x, y = coordinates
    neighbours = []
    # North, east, south, west
    if(x + 1 < gridSize):
        neighbours.append((x + 1, y))
    if(x - 1 >= 0):
        neighbours.append((x - 1, y))
    if (y + 1 < gridSize):
        neighbours.append((x, y + 1))
    if (y - 1 >= 0):
        neighbours.append((x, y - 1))
    # NW, NE, SW, SE
    if (x + 1 < gridSize and y + 1 < gridSize): #NE
        neighbours.append((x + 1, y + 1))
    if (x - 1 >= 0 and y + 1 < gridSize): #NW
        neighbours.append((x - 1, y + 1))
    if (x - 1 >= 0 and y - 1 >= 0):  # SW
        neighbours.append((x - 1, y - 1))
    if (x + 1 < gridSize and y - 1 >= 0): #SE
        neighbours.append((x + 1, y - 1))
    return neighbours

def generate_mnist_data():
    data_dir = '../data/mnist'
    mnist = input_data.read_data_sets(data_dir, one_hot=False)
    trainingSet = mnist.train
    labels = np.array([trainingSet.labels])
    concatenated = np.concatenate((trainingSet.images, labels.T), axis = 1)
    return concatenated, concatenated

def simple_plot(yvals,xvals=None,xtitle='X',ytitle='Y',title='Y = F(X)', label = ''):
    xvals = xvals if xvals is not None else list(range(len(yvals)))
    plt.plot(xvals,yvals, label = label)
    plt.pause(0.01)
    plt.xlabel(xtitle); plt.ylabel(ytitle); plt.title(title)
    plt.legend()
    plt.draw()
    plt.pause(.001)

def plot_training_history(error_hist, validation_hist=[], xtitle='Epoch', ytitle='Error', title='History', fig=True):
    # PLT.ion()
    if fig: plt.figure()
    if len(error_hist) > 0:
        simple_plot([p[1] for p in error_hist], [p[0] for p in error_hist], xtitle=xtitle, ytitle=ytitle, title=title,
                    label="Accuracy - [%]")
        # PLT.hold(True)
    if len(validation_hist) > 0:
        simple_plot([p[1] for p in validation_hist], [p[0] for p in validation_hist], xtitle=xtitle, ytitle=ytitle,
                    title=title, label="Validation loss")

