import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.decomposition import PCA

# Loading the csv into a data frame and creating attribute columns for normal
PointCloud = pd.read_csv('Input/SmallPointCloud.csv', names=['X', 'Y', 'Z'], dtype='float')
PointCloud['NX'] = pd.Series(0, index=PointCloud.index, dtype='float')
PointCloud['NY'] = pd.Series(0, index=PointCloud.index, dtype='float')
PointCloud['NZ'] = pd.Series(0, index=PointCloud.index, dtype='float')

# Finding K nearest neighbours : KNN
NbrsNum = 5
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

# Nodes : Write to CSV
PointCloud.to_csv('PointCloudWNormals.csv')
print("PointCloudWNormals.csv was saved.")