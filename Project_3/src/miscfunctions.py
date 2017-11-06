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
            vectorNorm = 1 #np.linalg.norm(np.array([x,y]))
            x_normed, y_normed = x/vectorNorm, y/vectorNorm
            cities[cityNmbr-1,:] = x_normed, y_normed
        max_x = np.max(cities[:, 0])
        min_x = np.min(cities[:, 0])
        max_y = np.max(cities[:, 1])
        min_y = np.min(cities[:, 1])
        cities[:, 0] = (cities[:, 0] - min_x) / (max_x - min_x)
        cities[:, 1] = (cities[:, 1] - min_y) / (max_y - min_y)
    return cities

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm
