import csv
import math
import sys
import numpy
import matplotlib.pyplot as plot


# colors array for scatter plot
colors = ['red', 'green', 'skyblue', 'gray', 'gold', 'orange', 'mediumpurple', 'violet', 'cyan', 'magenta']

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


# plot the data points in clusters in a scatter plot
def scatterplot(final_assignment, k, centroids):
    for i in range(k):
        x = numpy.zeros(len(final_assignment[i]) + 1)
        y = numpy.zeros(len(final_assignment[i]) + 1)
        scatter_colors = numpy.zeros(len(final_assignment[i]) + 1, dtype=object)
        for j in range(len(final_assignment[i])):
            x[j] = final_assignment[i][j][0]
            y[j] = final_assignment[i][j][1]
            scatter_colors[j] = colors[i]

        x[len(final_assignment[i])] = centroids[i][0]
        y[len(final_assignment[i])] = centroids[i][1]
        scatter_colors[len(final_assignment[i])] = 'black'
        # print(x, y)
        plot.scatter(x, y, c=scatter_colors)
    plot.show()


# find the error value by adding the distances of all the data points
def errorfunction(cluster_dist, centroid):
    sum = 0.0
    for i in range(len(cluster_dist)):
        for j in range(len(cluster_dist[i])):
            sum += math.pow(cluster_dist[i][j][0] - centroid[i][0], 2) + math.pow(cluster_dist[i][j][1] - centroid[i][1], 2)
    return sum


# algorithm
def k_means(data, k):
    #Initial centroid assignment. Randomly select centroids.
    cloned_data = numpy.copy(data)
    numpy.random.shuffle(cloned_data)
    centroids = numpy.zeros((k, 2))
    updated_centroids = numpy.zeros((k, 2))
    cluster_distribution = {}
    for i in range(k):
        centroids[i] = cloned_data[i]

    #Data assignment and updation
    while True:
        for i in range(k):
            cluster_distribution[i] = []

        # assign each data point to a cluster using the euclidean distance formula
        for d in data:
            s_i = sys.maxsize
            c_index = 0
            counter = 0
            for c in centroids:
                s_i_update = math.pow(d[0] - c[0], 2) + math.pow(d[1] - c[1], 2)
                if s_i_update < s_i:
                    s_i = s_i_update
                    c_index = counter
                counter += 1
            cluster_distribution[c_index].append(d)

        # update the centroids
        for i in range(k):
            x_val = 0.0
            y_val = 0.0
            for j in range(len(cluster_distribution[i])):
                x_val += cluster_distribution[i][j][0]
                y_val += cluster_distribution[i][j][1]

            x_val = x_val / len(cluster_distribution[i])
            y_val = y_val / len(cluster_distribution[i])
            updated_centroids[i][0] = x_val
            updated_centroids[i][1] = y_val

        # stop execution when cenroids do not change
        if (centroids == updated_centroids).all():
            errorvalue = errorfunction(cluster_distribution, centroids)
            # print(errorvalue, centroids)
            return errorvalue, cluster_distribution, centroids
        else:
            centroids = updated_centroids


def main():
    data = parsedata()
    finalErrorValues = []
    minClusterDist = {}
    finalPlotX = numpy.arange(2, 7)
    # execute the algorithm for k = 2, 3, 4, 5, 6
    for k in range(2, 7, 1):
        minErrorVal = sys.maxsize
        for i in range(10):
            errorvalue, cluster_distribution, centroids = k_means(data, k)
            # print(errorvalue)
            # get minimum error value for all the iterations
            if errorvalue < minErrorVal:
                minErrorVal = errorvalue
                minClusterDist = cluster_distribution
                min_centroids = centroids
        finalErrorValues.append(minErrorVal)
        finalPlot = numpy.array(finalErrorValues)
        print('K: ' + str(k) + ' WCSS: ' + str(minErrorVal))
        # print(min_centroids)
        scatterplot(minClusterDist, k, min_centroids)

    # plot the final error graph
    plot.plot(finalPlotX, finalPlot)
    plot.xlabel("K")
    plot.ylabel("Error")
    plot.show()


main()