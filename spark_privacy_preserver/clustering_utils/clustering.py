from .. import gv
from .distance_calculation import Calculator as cl

class Clustering:

    @staticmethod
    def level_cluster(df,cluster_num):
        cluster_len = df.loc[df['cluster_number']==cluster_num].shape[0]
        additional_entries = df.loc[df['cluster_number']==cluster_num].nlargest(cluster_len-gv.k,'cluster_distance')
        df.at[additional_entries.index,'cluster_number'] = -1
        df.at[additional_entries.index,'cluster_distance'] = -1 

    @staticmethod
    def adjust_big_clusters1(df):
        temp = df['cluster_number'].value_counts()
        big_clusters = temp.loc[temp > gv.k].index
        for index in big_clusters:
            Clustering.level_cluster(df,index)


    @staticmethod
    def find_best_cluster(df,row,clusters,factor= 20):
        """
        calcuate the distance from a data point to all the cluster centroids
        and find the best cluster
        
        row - datapoint as pandas series from the dataframe, 
        clusters - cluster dataframe with all the cluster centroids
        factor - is to change priority between numerical data points and categorical data points
        when you give a higher value for the factor numerical data points get higher priority 
        resulting lower ranges for a cluster. but this cause higher data loss for categorical data

        returns the index/cluster_number that best fit to the given row.
        """

        numerical_col = clusters[gv.GV['NUM_COL']]
        categorical_col = clusters[gv.GV['CAT_COL']]
        ranges = numerical_col.max() - numerical_col.min()
        numerical_distance = cl.cal_num_col_dist(row[gv.GV['NUM_COL']],numerical_col,ranges,factor)
        categorical_distance = cl.cal_cat_col_dist3(row,categorical_col)
        df.at[row.name,'cluster_distance'] = (numerical_distance.sum(axis=1)+categorical_distance).min()
        return (numerical_distance.sum(axis=1)+categorical_distance).idxmin()

    
