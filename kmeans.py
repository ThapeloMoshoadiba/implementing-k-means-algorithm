'''

Data Science, Algorithms and Advanced Software Engineering
Task 17
Thapelo Moshoadiba, 17 October 2019

● Your task is to implement the k-means algorithm and run it on the provided data. Open the file ‘kmeans.py’ and fill in the missing code to implement the algorithm -
    comments have been placed to guide you. Also take a look at the data provided which consists of the two variables life expectancy, birth rate) measured for each
    country, and there is one dataset for 2008 and one from 1953. This data is in a csv file, which can be opened just like a text file. Open each file and take a look
    at how the data is laid out. Your algorithm should work on either data set.
● To compute the mean (or average) of a number of observations, you simply add all the observations together and then divide by the number of observations you added
    together. The two-dimensional mean is no different. You have a number of x and y values, so to compute the mean
● (x, y) point of a number of points, you simply compute the mean of all the x values, and the mean of all the y values.
● You should let the user decide how many clusters there should be (although, if you are unsure, it might make sense to begin with just two clusters, and generalize
    later). The algorithm will need to run for a user-specified number of iterations, and output:
    1. The number of countries belonging to each cluster
    2. The list of countries belonging to each cluster
    3. The mean Life Expectancy and Birth Rate for each cluster
● How many iterations do you need to use? Well, that depends on the data, and the number of clusters chosen. How can we tell if the algorithm has converged? To get
    into this, we should talk more about convergence and “optimization”. Optimization, very broadly, is taking some measure of goodness, and making it as good as
    possible. The measure of goodness is called an “objective function”, and the value of this objective function depends on both the data and on the parameters (which
    are just the cluster means, in this case). The k-means algorithm is actually solving an optimization problem, where each iteration minimizes the value of an
    objective function. In this case, the objective function is the sum of squared distances between each point and the cluster center that each point belongs to.
    “Convergence” refers to the fact that the changes in the value of the objective function get smaller and smaller until eventually, the value appears to be the same
    from one iteration to the next, and we can say that the algorithm has converged.

● In your main iteration loop, for each data point, calculate the squared distance between each point and the cluster mean to which it belongs, and sum all of these
	squared distances.
● Print out this sum once each iteration, and you can watch the objective function converge. Make sure you are picking enough iterations to allow the algorithm to
	converge. If the value of the objective function gets worse, then you either have a bug in your k-means algorithm or a bug in your objective function
	calculation (or both!). Thus, explicitly calculating the objective function also serves to test your code. We call this a “sanity check”.
● There are three provided datasets: data1953.csv, data2008.csv, and dataBoth.csv. The first two contain 197 countries, with Life Expectancy and Birth Rate measured for
	each country, but the first has these measurements taken from 1953, and the second from 2008. dataBoth. csv contains both the 2008 values and the 1953 values,
	but pretending that the countries from different years are different countries. Thus, we can see, for example, that (1953)Zimbabwe falls in the same cluster as
	(2008)Zimbabwe, but that (1953)Botswana and (2008)Botswana fall into different clusters.

Task 17.3

Create a text file called ‘interpretation.txt’ and fill in your answers to the questions below. Run k-means using 3 clusters on the 1953 and 2008 datasets separately.
What do you see? Take note of how the clusters change from 1953 to 2008. You will need to pay attention not only to which countries are in clusters together but also
to the Life Expectancy and BirthRates for those clusters. Next, run the algorithm with 4 clusters on dataBoth.csv. Note any observations in your text file. Which
countries are moving up clusters? How does (2008)South Africa” compare to “(1953)United States”? Are there any 2008 countries that are in a cluster that is made up
mostly of 1953 countries? Try and explain why. Are there any 1953 countries that are in a cluster that is made up of mostly 2008 countries? Try and explain why in your
text file.

'''

import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

def data1985 ():
    data = pd.read_csv('data1953.csv')
    print (data.head()) # .head() is a pandas method that is used to return top n (5 by default) rows of a data frame or series. Just to show the layout and type of information

    # we are selecting the variables "Countries", "BirthRate(Per1000 - 1953)" and LifeExpectancy(1953)

    X = data[["Countries","BirthRate(Per1000 - 1953)","LifeExpectancy(1953)"]]


    # choosing the number of clusters (k) and selecting random centroids for each cluster

    K = int (input ("\nEnter the no. of clusters you want (between 1 and 7): \n"))

    Centroids = (X.sample(n=K))

    iteration = int (input ("\nEnter the no. of iterations you want the algorithm to run for: \n"))
    
    # Assign all the points to the closest cluster centroid. Recompute centroids of newly formed clusters. Repeat

    diff = 1
    j=0
    limiter = 0

    print ("\nNow we monitor convergence:")
    while (limiter <= iteration):

        limiter += 1
        
        XD=X
        i=1
        for index1,row_c in Centroids.iterrows():
            ED=[]
            for index2,row_d in XD.iterrows():
                d1=(row_c["BirthRate(Per1000 - 1953)"]-row_d["BirthRate(Per1000 - 1953)"])**2
                d2=(row_c["LifeExpectancy(1953)"]-row_d["LifeExpectancy(1953)"])**2
                d=np.sqrt(d1+d2)
                ED.append(d)
            X[i]=ED
            i=i+1

        C=[]
        for index,row in X.iterrows():
            min_dist=row[1]
            pos=1
            for i in range(K):
                if row[i+1] < min_dist:
                    min_dist = row[i+1]
                    pos=i+1
            C.append(pos)
        X["Cluster"]=C
        Centroids_new = X.groupby(["Cluster"]).mean()[["LifeExpectancy(1953)","BirthRate(Per1000 - 1953)"]]
        if j == 0:
            diff=1
            j=j+1
        else:
            diff = (Centroids_new['LifeExpectancy(1953)'] - Centroids['LifeExpectancy(1953)']).sum() + (Centroids_new['BirthRate(Per1000 - 1953)'] - Centroids['BirthRate(Per1000 - 1953)']).sum()
            print(diff.sum())
        Centroids = X.groupby(["Cluster"]).mean()[["LifeExpectancy(1953)","BirthRate(Per1000 - 1953)"]]

    color=['blue','green','cyan', 'yellow', 'purple', 'brown', 'orange'] # this limits the number of possible clusters to 7
    for k in range(K):
        data=X[X["Cluster"]==k+1]
        print ("\nNumber of countries in cluster " + str(k+1))
        print (len(data["Countries"]))
        print ("List of countries in cluster " + str(k+1))
        print (data["Countries"])
        plt.scatter(data["BirthRate(Per1000 - 1953)"],data["LifeExpectancy(1953)"],c=color[k])
    plt.scatter(Centroids["BirthRate(Per1000 - 1953)"],Centroids["LifeExpectancy(1953)"],c='red', marker = 'X')

    print ("\nThe means for the clusters are as follows: ")
    print(Centroids)

    plt.xlabel('Birth Rate')
    plt.ylabel('Life Expectancy')
    plt.show()


def data2008 ():
    data = pd.read_csv('data2008.csv')
    print (data.head()) # .head() is a pandas method that is used to return top n (5 by default) rows of a data frame or series. Just to show the layout and type of information

    # we are selecting the variables "Countries", "BirthRate(Per1000 - 2008)" and LifeExpectancy(2008)

    X = data[["Countries","BirthRate(Per1000 - 2008)","LifeExpectancy(2008)"]]


    # choosing the number of clusters (k) and selecting random centroids for each cluster

    K = int (input ("\nEnter the no. of clusters you want (between 1 and 7): \n"))

    Centroids = (X.sample(n=K))

    iteration = int (input ("\nEnter the no. of iterations you want the algorithm to run for: \n"))
    
    # Assign all the points to the closest cluster centroid. Recompute centroids of newly formed clusters. Repeat

    diff = 1
    j=0
    limiter = 0

    print ("\nNow we monitor convergence:")
    while (limiter <= iteration):

        limiter += 1
        
        XD=X
        i=1
        for index1,row_c in Centroids.iterrows():
            ED=[]
            for index2,row_d in XD.iterrows():
                d1=(row_c["BirthRate(Per1000 - 2008)"]-row_d["BirthRate(Per1000 - 2008)"])**2
                d2=(row_c["LifeExpectancy(2008)"]-row_d["LifeExpectancy(2008)"])**2
                d=np.sqrt(d1+d2)
                ED.append(d)
            X[i]=ED
            i=i+1

        C=[]
        for index,row in X.iterrows():
            min_dist=row[1]
            pos=1
            for i in range(K):
                if row[i+1] < min_dist:
                    min_dist = row[i+1]
                    pos=i+1
            C.append(pos)
        X["Cluster"]=C
        Centroids_new = X.groupby(["Cluster"]).mean()[["LifeExpectancy(2008)","BirthRate(Per1000 - 2008)"]]
        if j == 0:
            diff=1
            j=j+1
        else:
            diff = (Centroids_new['LifeExpectancy(2008)'] - Centroids['LifeExpectancy(2008)']).sum() + (Centroids_new['BirthRate(Per1000 - 2008)'] - Centroids['BirthRate(Per1000 - 2008)']).sum()
            print(diff.sum())
        Centroids = X.groupby(["Cluster"]).mean()[["LifeExpectancy(2008)","BirthRate(Per1000 - 2008)"]]

    color=['blue','green','cyan', 'yellow', 'purple', 'brown', 'orange'] # this limits the number of possible clusters to 7
    for k in range(K):
        data=X[X["Cluster"]==k+1]
        print ("\nNumber of countries in cluster " + str(k+1))
        print (len(data["Countries"]))
        print ("List of countries in cluster " + str(k+1))
        print (data["Countries"])
        plt.scatter(data["BirthRate(Per1000 - 2008)"],data["LifeExpectancy(2008)"],c=color[k])
    plt.scatter(Centroids["BirthRate(Per1000 - 2008)"],Centroids["LifeExpectancy(2008)"],c='red', marker = 'X')

    print ("\nThe means for the clusters are as follows: ")
    print(Centroids)

    plt.xlabel('Birth Rate')
    plt.ylabel('Life Expectancy')
    plt.show()


def dataBoth ():
    data = pd.read_csv('dataBoth.csv')
    print (data.head()) # .head() is a pandas method that is used to return top n (5 by default) rows of a data frame or series. Just to show the layout and type of information

    # we are selecting the variables "Countries", "BirthRate(Per1000)" and LifeExpectancy

    X = data[["Countries","BirthRate(Per1000)","LifeExpectancy"]]


    # choosing the number of clusters (k) and selecting random centroids for each cluster

    K = int (input ("\nEnter the no. of clusters you want (between 1 and 7): \n"))

    Centroids = (X.sample(n=K))

    iteration = int (input ("\nEnter the no. of iterations you want the algorithm to run for: \n"))
    
    # Assign all the points to the closest cluster centroid. Recompute centroids of newly formed clusters. Repeat

    diff = 1
    j=0
    limiter = 0

    print ("\nNow we monitor convergence:")
    while (limiter <= iteration):

        limiter += 1
        
        XD=X
        i=1
        for index1,row_c in Centroids.iterrows():
            ED=[]
            for index2,row_d in XD.iterrows():
                d1=(row_c["BirthRate(Per1000)"]-row_d["BirthRate(Per1000)"])**2
                d2=(row_c["LifeExpectancy"]-row_d["LifeExpectancy"])**2
                d=np.sqrt(d1+d2)
                ED.append(d)
            X[i]=ED
            i=i+1

        C=[]
        for index,row in X.iterrows():
            min_dist=row[1]
            pos=1
            for i in range(K):
                if row[i+1] < min_dist:
                    min_dist = row[i+1]
                    pos=i+1
            C.append(pos)
        X["Cluster"]=C
        Centroids_new = X.groupby(["Cluster"]).mean()[["LifeExpectancy","BirthRate(Per1000)"]]
        if j == 0:
            diff=1
            j=j+1
        else:
            diff = (Centroids_new['LifeExpectancy'] - Centroids['LifeExpectancy']).sum() + (Centroids_new['BirthRate(Per1000)'] - Centroids['BirthRate(Per1000)']).sum()
            print(diff.sum())
        Centroids = X.groupby(["Cluster"]).mean()[["LifeExpectancy","BirthRate(Per1000)"]]

    color=['blue','green','cyan', 'yellow', 'purple', 'brown', 'orange'] # this limits the number of possible clusters to 7
    for k in range(K):
        data=X[X["Cluster"]==k+1]
        print ("\nNumber of countries in cluster " + str(k+1))
        print (len(data["Countries"]))
        print ("List of countries in cluster " + str(k+1))
        print (data["Countries"])
        plt.scatter(data["BirthRate(Per1000)"],data["LifeExpectancy"],c=color[k])
    plt.scatter(Centroids["BirthRate(Per1000)"],Centroids["LifeExpectancy"],c='red', marker = 'X')

    print ("\nThe means for the clusters are as follows: ")
    print(Centroids)

    plt.xlabel('Birth Rate')
    plt.ylabel('Life Expectancy')
    plt.show()


def file (x):

    if x == 1:
        data1985 ()
    elif x == 2:
        data2008 ()
    elif x == 3:
        dataBoth ()

print ("Which file would you like to open:"
       "\n1. data1985"
       "\n2. data2008"
       "\n3. dataBoth")

num = int (input ("\nEnter the number: "))

file(num)
