import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sb
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.preprocessing
import scipy.cluster.hierarchy as sch
from fcmeans import FCM
import skfuzzy as fuzz
import pylab
import sklearn.mixture as mixture
import pyclustertend 
import random
import matplotlib.cm as cm
from reader import reader as Reader

class main(object):

    def __init__(self, csvDoc):
        # Universal Doc
        self.csvDoc = csvDoc
        # Classes
        R = Reader(csvDoc)
        self.df = R.movies

    def prep(self, dataset):
        df = dataset
        df = df.drop(columns=[
            'originalTitle', 'title', 'director',
            'homePage', 'actors', 'actorsCharacter'
            ])
        return df

    def groupBy_MainGenre(self):
        df = self.df
        df = self.prep(df)
        df['genres'] = df['genres'].str.split('|')
        df = df.explode('genres').reset_index(drop=True).drop_duplicates(subset=['id'])
        df = df.drop(columns=['id'])

        #df.drop(['genres'],1).hist()
        #plt.show()
        #df = df.groupby('genres').size().sort_values(ascending=False)
        #print(df)
        #df.groupby('genres').size()
        return df
    
    def hopkins(self):
        df = self.groupBy_MainGenre()
        self.X = np.array(df[[
            'budget','revenue','runtime',
            'voteCount', 'voteAvg','actorsAmount'
        ]])
        X = self.X
        self.Y = np.array(df[['genres']])
        random.seed(10000)
        X_scale = sklearn.preprocessing.scale(X)
        hop = pyclustertend.hopkins(X,len(X))

        return hop, X_scale, X

    def vat(self):
        hop, X_scale = self.hopkins()
        pyclustertend.vat(X_scale)
        vat = pyclustertend.vat(self.X)
        return vat

    def clusterNum(self):
        hop, X_scale, X = self.hopkins()
        numeroClusters = range(1,11)
        wcss = []
        for i in numeroClusters:
            kmeans = cluster.KMeans(n_clusters=i)
            kmeans.fit(X_scale)
            wcss.append(kmeans.inertia_)

        # plt.plot(numeroClusters, wcss)
        # plt.xlabel("Número de clusters")
        # plt.ylabel("Score")
        # plt.title("Gráfico de Codo")
        # plt.show()


        
    # fuzzy c-means algorithms 
    def fuzzy_cMeans(self):
        hop, X_scale, X = self.hopkins()
        
        fcm = FCM(n_clusters=4)
        fcm.fit(X)

        fcm_centers = fcm.centers
        fcm_labels = fcm.predict(X)

        plt.title("Grouping by Fuzzy C-Means")

        plt.scatter(X[:,0],X[:,1], c=fcm_labels, cmap='plasma')
        plt.scatter(fcm_centers[:,0],fcm_centers[:,1], c='green', marker='v')
        plt.show()

    # 
    def mix_gaussians(self):
        hop, X_scale, X = self.hopkins()

        gmm = mixture.GaussianMixture(n_components = 4).fit(X)
        labels = gmm.predict(X)

        plt.title("Grouping by Mixture of Gaussians")
        plt.scatter(X[:, 0], X[:, 1], c=labels,cmap="plasma")
        plt.show()

    def hierarchical_clustering(self):
        
        hop, X_scale, X = self.hopkins()
        # dendograma = sch.dendrogram(sch.linkage(X, method='ward'))
        hc = cluster.AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
        
        movieHC = hc.fit_predict(X)
        plt.title("Grouping by Hierarchical clustering")
        plt.scatter(X[:, 0], X[:, 1], c=movieHC, cmap="plasma")
        plt.show()



    def silhouette_method_FCM(self):
        hop, X_scale, X = self.hopkins()
        range_n_clusters = [2, 3, 4]

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            clusterer = FCM(n_clusters=n_clusters)
            
            clusterer.fit(X)
            cluster_labels = clusterer.predict(X)

            # The silhouette_score gives the average value for all the samples.

            silhouette_avg = metrics.silhouette_score(X, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = metrics.silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(
                X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
            )

            # Labeling the clusters
            centers = clusterer.centers
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for Fuzzy C-Means clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )

        plt.show()  




    def silhouette_method_HC(self):
        hop, X_scale, X = self.hopkins()
        range_n_clusters = [2, 3, 4]

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])


            clusterer = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            silhouette_avg = metrics.silhouette_score(X, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = metrics.silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(
                X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
            )

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for Hierarchical clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )

        plt.show()  


        
    def silhouette_method_gmm(self):
        hop, X_scale, X = self.hopkins()
        range_n_clusters = [2, 3, 4]

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])


            clusterer = mixture.GaussianMixture(n_components=n_clusters).fit(X)
            cluster_labels = clusterer.fit_predict(X)


            # The silhouette_score gives the average value for all the samples.
            silhouette_avg = metrics.silhouette_score(X, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = metrics.silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(
                X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
            )

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for Mixture of Gaussians clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )

        plt.show() 

        

m = main('movies.csv')
m.groupBy_MainGenre()
m.hopkins()
m.clusterNum()
m.silhouette_method_HC()
#m.vat()
