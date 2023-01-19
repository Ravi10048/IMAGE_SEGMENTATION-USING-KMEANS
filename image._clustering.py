import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
img_mat=mpimg.imread("flower.jpg")
img_mat1=img_mat.copy()
print(img_mat.shape)
print(img_mat)
img_plot=plt.imshow(img_mat)
plt.show()

############################333
# # resize the image using PIL
# img=Image.open("cell.jpg")
# img_res=img.resize((300,200))
# img_res.save("cell_resize.jpg")

# img_mat=mpimg.imread("cell_resize.jpg")
# print(img_mat.shape)
# print(img_mat)
# img_plot=plt.imshow(img_mat)
# plt.show()


Blue = img_mat[:,:,0][0]; Green = img_mat[:,:,1][0]; Red = img_mat[:,:,2][0]

# print(Red,Green,Blue)



df=pd.DataFrame({"Red": Red,"Green": Green,"Blue": Blue})
df.to_csv("cell_data.csv", sep='\t')
print(df)
X = df.iloc[:,:].values 
# print(X)

'''we did not know the suitable number of clusters and these varies with dataset 
so we use a parameter to choose suitable no. of clusters knows as wcss- with cluster sum of square'''

# finding wcss value for different number of clusters

wcss = [] #wcss- with cluster sum of square

for i in range(1,11):  # tends to find distance between each data point to a centroid (cluster point) and each circle have each own centroid
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42) # 
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)  # give and append wcss value

# We assume first 2 data as centroid then find the distance between them by all centroid then update the centroid then again continue

# plot elbow graph
sns.set()
plt.plot(range(1,11),wcss)
plt.title("Elbow point graph")
plt.xlabel("Number of cluster")
plt.ylabel("WCSS value")
plt.show()  # we will consider k value we gives us sharp significant drop in value


'''Optimum Number of Clusters = 3'''

#Training the k-Means Clustering Model

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

print(Y)

'''3 Clusters - 0, 1, 2,3
Visualizing all the Clusters'''

# plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))  # X[Y==0,0]  1st 0 represent cluster no. and 2nd represent X[0]
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')  # centroid 1 is plotted
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2') # centroid 2 is plotted
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5') # s = size c = color
# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centroids')

plt.title('flower plotting')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
# u=[X[Y==0,0],X[Y==0,1],X[Y==0,2]]
red_in=[]
green_in=[]
blue_in=[]
bg_in=[]
for i in range(len(Y)):
    if(Y[i]==0):
        red_in.append(i)
    elif(Y[i]==1):
        green_in.append(i)
    elif(Y[i]==2):
        blue_in.append(i) 
    elif(Y[i]==3):
        bg_in.append(i)      

for i in range(len(blue_in)):
    if(i in blue_in):
        pass
    else:
        img_mat1[ :,:,1][0][i]=0
        img_mat1[ :,:,1][1][i]=0
        img_mat1[ :,:,1][2][i]=0
print(img_mat1[:,:,1][0])   
Red=img_mat1[:,:,1]     
# print(Red)

img_plot=plt.imshow(Red)
plt.show()