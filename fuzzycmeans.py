import csv
import math
import sys
import numpy
import matplotlib.pyplot as plot


# parsed data to construct a numpy array
def parsedata():
    filename = '545_cluster_dataset programming 3.txt'
    input_data = []
    with open(filename, newline='') as datafile:
        cluster_data = csv.reader(datafile, delimiter=' ')
        for row in cluster_data:
            inputdatarow = []
            for column in row:
                if len(column) > 0:
                    inputdatarow.append(float(column))
            input_data.append(inputdatarow)
    final_cluster_data = numpy.array(input_data)
    return final_cluster_data


# plot the clusters using scatter plot
def scatterplot(final_assignment, c, centroids):
    for i in range(c):
        x = numpy.zeros(len(final_assignment[i]))
        y = numpy.zeros(len(final_assignment[i]))
        for j in range(len(final_assignment[i])):
            x[j] = final_assignment[i][j][0]
            y[j] = final_assignment[i][j][1]

        # print(x, y)
        plot.scatter(x, y)
    plot.show()


# calculate the WCSS values
def errorfunction(data, centroid, weights, m):
    sum = 0.0
    for i in range(len(data)):
        for j in range(len(centroid)):
            sum += (math.pow(data[i][0] - centroid[j][0], 2) + math.pow(data[i][1] - centroid[j][1], 2)) * math.pow(weights[i][j], m)
    return sum


def fuzzyc_means(data, c, m):
    #Initial probability assignment
    weights = numpy.random.dirichlet(numpy.ones(c), size=len(data))
    centroids = numpy.zeros((c, 2))
    updated_centroids = numpy.zeros((c, 2))

    # run until the algorithm has converged
    while True:
        # compute the centroids using the weight assignments
        powered_weights = weights ** m
        cluster_weights = numpy.sum(powered_weights, axis=0)

        for i in range(c):
            cloned_data = numpy.copy(data)
            for j in range(len(cloned_data)):
                cloned_data[j][0] *= powered_weights[j][i]
                cloned_data[j][1] *= powered_weights[j][i]
            sum_data = numpy.sum(cloned_data, axis=0)
            updated_centroids[i] = sum_data / cluster_weights[i]
        # print(centroids)

        # compute the distance of each centroid to the data point
        distances = numpy.zeros((len(data), c))
        for i in range(c):
            for j in range(len(data)):
                distances[j][i] = math.pow(data[j][0] - updated_centroids[i][0], 2) + math.pow(data[j][1] - updated_centroids[i][1], 2)
        # print(distances)

        # compute the coefficient membership grade for each data point
        modified_distances = numpy.zeros((len(data), c))
        for i in range(c):
            for j in range(len(data)):
                modified_distances[j][i] = math.pow(1 / distances[j][i], (2 / (m - 1)))
        # print(modified_distances)

        # update the weights for each data point
        sum_distances = numpy.sum(modified_distances, axis=1)
        for i in range(c):
            for j in range(len(data)):
                weights[j][i] = modified_distances[j][i] / sum_distances[j]
        # print(weights)

        # stop the computation when the centroids do not change
        if (centroids == updated_centroids).all():
            errorvalue = errorfunction(data, centroids, weights, m)
            return errorvalue, centroids, weights
        else:
            centroids = updated_centroids


# created clusters to plot on the scatter plot
def assign_cluster(data, weights, c):
    cluster_dist = {}
    max_prob = numpy.argmax(weights, axis=1)
    for i in range(c):
        cluster_dist[i] = []
    for i in range(len(data)):
        cluster_dist[max_prob[i]].append(data[i])
    return cluster_dist


def main():
    # get the data
    data = parsedata()
    finalErrorValues = []
    finalPlotX = numpy.arange(2, 7)
    # set the fuzzifier value
    m = 2
    # execute the algorithm for c = 2, 3, 4, 5, 6
    for c in range(2, 7, 1):
        minErrorVal = sys.maxsize
        for i in range(10):
            errorvalue, centroids, weights = fuzzyc_means(data, c, m)
            # print(errorvalue)
            # get the minimum error value
            if errorvalue < minErrorVal:
                minErrorVal = errorvalue
                min_centroids = centroids
                final_weights = weights
        finalErrorValues.append(minErrorVal)
        finalPlot = numpy.array(finalErrorValues)
        print('C: ' + str(c) + ' WCSS: ' + str(minErrorVal))
        # print(min_centroids)
        finalClusterDist = assign_cluster(data, final_weights, c)
        scatterplot(finalClusterDist, c, min_centroids)

    # plot the final error values
    plot.plot(finalPlotX, finalPlot)
    plot.xlabel("C")
    plot.ylabel("Error")
    plot.show()


main()