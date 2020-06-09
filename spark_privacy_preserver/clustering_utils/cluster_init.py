from .. import gv
import numpy as np
class ClusterInit:

    @staticmethod
    def find_best_cluster_gens(n_clusters,dataframe):
        
        length = len(dataframe)
        probs = [dataframe[column].value_counts()/length  for column in gv.GV['QI']]
        dataframe['probability_sum'] = dataframe[gv.GV['QI']].apply(lambda row: np.sum([probs[i][row[column]] for i,column in enumerate(gv.GV['QI'])]),axis = 1)
        points = dataframe.sample(n = n_clusters , weights = 'probability_sum')
        dataframe = dataframe[gv.GV['QI']]
        return points[gv.GV['QI']]

    @staticmethod
    def select_centroids_using_weighted_column(n_clusters,dataframe):
        counts = [len(dataframe[column].unique())  for column in gv.GV['QI']]
        column_counts = sorted(counts)
        column = counts.index(column_counts[-1])
        try:
          return dataframe.sample(n = n_clusters , weights = gv.GV['QI'][column])
        except Exception:
          return ClusterInit.find_best_cluster_gens(n_clusters,dataframe)
        

    @staticmethod
    def random_sample_centroids(n_clusters,unique_rows):
        return unique_rows.sample(n=n_clusters)