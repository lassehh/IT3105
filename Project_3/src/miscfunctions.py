import numpy as np


def generate_tsp_data(problemNumbr):
    with open('../data/TSP_Problems/' + str(problemNumbr) + '.txt', 'r') as f:
        firstLine = f.readline().strip("\n")
        firstLine = firstLine.split(":")
        nmbrOfCities = firstLine[1]
        cities = np.zeros((int(nmbrOfCities), 2))
        f.__next__()
        for line in f:
            line = line.strip("\n")
            if(line == "EOF"):
                break
            line = line.split(" ")
            cityNmbr, x, y = line
            cityNmbr, x, y = int(cityNmbr), int(x), int(y)
            vectorNorm = np.linalg.norm(np.array([x,y]))
            x_normed, y_normed = x/vectorNorm, y/vectorNorm
            cities[cityNmbr-1,:] = x_normed, y_normed
    return cities


