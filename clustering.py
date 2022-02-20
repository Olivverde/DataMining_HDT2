import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sb
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.preprocessing
import scipy.cluster.hierarchy as sch
import skfuzzy as fuzz
import pylab
import sklearn.mixture as mixture
import pyclustertend 
import random
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

        return hop, X_scale

    def vat(self):
        hop, X_scale = self.hopkins()
        pyclustertend.vat(X_scale)
        vat = pyclustertend.vat(self.X)
        return vat

    def clusterNum(self):
        hop, X_scale = self.hopkins()
        numeroClusters = range(1,11)
        wcss = []
        for i in numeroClusters:
            kmeans = cluster.KMeans(n_clusters=i)
            kmeans.fit(X_scale)
            wcss.append(kmeans.inertia_)

        plt.plot(numeroClusters, wcss)
        plt.xlabel("Número de clusters")
        plt.ylabel("Score")
        plt.title("Gráfico de Codo")
        plt.show()

m = main('movies.csv')
m.groupBy_MainGenre()
m.hopkins()
m.clusterNum()
#m.vat()
