import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.decomposition import PCA
import networkx as nx
import csv

# Iteration Limiter ( current dataframe has 4357 rows )
limiter = 20000

# Loading the csv into a data frame and creating attribute columns for normal
PointCloud = pd.read_csv('Input/SmallPointCloud.csv', names=['X', 'Y', 'Z'], dtype='float')
PointCloud['NX'] = pd.Series(0, index=PointCloud.index, dtype='float')
PointCloud['NY'] = pd.Series(0, index=PointCloud.index, dtype='float')
PointCloud['NZ'] = pd.Series(0, index=PointCloud.index, dtype='float')
PointCloud['C'] = pd.Series(0, index=PointCloud.index, dtype='float')

# Finding K nearest neighbours : KNN
NbrsNum = 10
nbrs = NearestNeighbors(n_neighbors=NbrsNum, algorithm='ball_tree').fit(PointCloud)
NbrsDistances, NbrsIndices = nbrs.kneighbors(PointCloud)

# Finding Neighbours in radius
# NbrsRad = 1.5
# nbrs = NearestNeighbors(radius=NbrsRad).fit(PointCloud)
# NbrsDistances, NbrsIndices = nbrs.radius_neighbors(PointCloud)

# loop goes through all of points in PointCloud and calculates normal vector from corresponding neighborhood
for i in range(len(NbrsIndices)):
    # list of list that contains XYZ coord. for all neighbours
    XYZ_neighborhood = []
    # list of IDs of neighbours
    ID_neighborhood = NbrsIndices[i]

    # loop goes through all neighbours and retrieves their XYZ coord.
    for j in range(0,len(ID_neighborhood)):
        XYZ_neighbour = []
        ID_neighbour = ID_neighborhood[j]
        X_Neighbour = PointCloud['X'][ID_neighbour]
        Y_Neighbour = PointCloud['Y'][ID_neighbour]
        Z_Neighbour = PointCloud['Z'][ID_neighbour]
        XYZ_neighbour.append(X_Neighbour)
        XYZ_neighbour.append(Y_Neighbour)
        XYZ_neighbour.append(Z_Neighbour)

        XYZ_neighborhood.append(XYZ_neighbour)

    # X is the input for pca method, it demands dataframe of np.array
    # More to how I understand these method here:
    #   https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-vectors-from-sklearn-pca
    #   http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    X = np.array(XYZ_neighborhood)
    # print (X)
    # n_components means "3D"
    pca = PCA(n_components=3)
    pca.fit(X)

    # I am not sure what the next line does. I commented it and seems to still work...
    # PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,svd_solver='auto', tol=0.0, whiten=False)

    # retrieving eigenValues/Vectors and finding smallest eigenValue and corresponding eigenVector
    eigenVectors = pca.components_
    eigenValues = pca.explained_variance_
    MinEigenValue,IDofMinEigeValue = min((eigenValues[i],i) for i in range(len(eigenValues)))
    normal = eigenVectors[IDofMinEigeValue]


    # saving to PointCloud
    PointCloud['NX'][i] = normal[0]
    PointCloud['NY'][i] = normal[1]
    PointCloud['NZ'][i] = normal[2]

    #PointCloud['C'][i] = MinEigenValue/eigenValues.sum()
    PointCloud['C'][i] =  MinEigenValue/sum(eigenValues)

for index, row in PointCloud.iterrows():
    high_point_ref = [0,0,200]

    #current point normal
    iNormal = [PointCloud['NX'][index],PointCloud['NY'][index],PointCloud['NZ'][index]]

    if np.dot(iNormal,high_point_ref) < 0:  # high_point_ref is just a very high point to make sure normals all flip

        PointCloud['NX'][index] *= -1
        PointCloud['NY'][index] *= -1
        PointCloud['NZ'][index] *= -1




########################################################################################################################

# Initializing the whole Graph or "Mother Graph"

MotherGraph = nx.Graph()

counter = 0
for index, row in PointCloud.iterrows():

    # Limiting the iteration so it doest take too long, set it to a high number if you want to run everything
    if counter > limiter : break
    counter += 1

    # Adding Nodes
    MotherGraph.add_node(index, C = PointCloud['C'][index]) # C is curvature

    # Here we iterate through the neighbourlist
    for j in range(NbrsIndices[index].size):
        # Adding Edges
        nested_index = NbrsIndices[index][j]
        # Calculating Curvature Product
        CP = PointCloud['C'][index] + PointCloud['C'][nested_index]
        NDP = abs(np.dot([PointCloud['NX'][index],PointCloud['NY'][index],PointCloud['NZ'][index]],
                         [PointCloud['NX'][nested_index],PointCloud['NY'][nested_index],PointCloud['NZ'][nested_index]]))
        MotherGraph.add_edge(index, nested_index, CP = CP, NDP = NDP)

EdgeList = list(MotherGraph.edges.data('CP'))

# Reporting
print "Graph : Calculated"

# Nodes : Write to CSV
PointCloud.to_csv('Output/Nodes.csv')

# Edges : Write to CSV
with open('Output/Edges.csv', 'wt') as csvfile:  ## writing the whole deference phase
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for p in range(len(EdgeList)):
        writer.writerow(EdgeList[p])

#print EdgeList

# NodesWithNormals : Write to CSV
PointCloud.to_csv('PointCloudWNormals.csv')
print("PointCloudWNormals.csv was saved.")
