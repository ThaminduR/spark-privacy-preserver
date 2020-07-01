import numpy as np
from .. import gv

class Calculator:

    @staticmethod
    def cal_num_col_dist(record,cluster_record,range_value,factor):
        '''
        This method is used to find the numerical distance between records and cluster genaralization values
        record :-  The record need to find the distance with cluster  type :- Pandas Series
        cluster_record :- Numerical generalization values of the clusters   type :- Pandas DataFrame
        range_value :- Clusters range values   type :- Pandas DataFrame
        factor :- factor need to multiply. Recommended 20. type:- Integer
        '''
        return abs((cluster_record - record)/range_value*factor)

    @staticmethod
    def cal_cat_col_dist1(row,categorical_col):
        '''
        This method is used to find the categorical distance between records and cluster genaralization values
        row :-  The record need to find the distance with cluster  type :- Pandas Series
        categorical_col :- Catergorical generalization values of the clusters   type :- Pandas DataFrame
        '''
        return (len(gv.GV['CAT_COL'])-np.where(
            row[gv.GV['CAT_COL']]==categorical_col,True,np.where(
                categorical_col == "*****",True,False)).sum(axis=1))*10
    @staticmethod
    def cal_cat_col_dist2(row,categorical_col):
        '''
        This method is used to find the categorical distance between records and cluster genaralization values
        row :-  The record need to find the distance with cluster  type :- Pandas Series
        categorical_col :- Catergorical generalization values of the clusters   type :- Pandas DataFrame
        '''
        return (len(gv.GV['CAT_COL'])-np.where(
            row[gv.GV['CAT_COL']]==categorical_col,1,np.where(
                categorical_col == "*****",0.5,0)).sum(axis=1))*10

    @staticmethod
    def cal_cat_col_dist3(row,categorical_col):
        '''
        This method is used to find the categorical distance between records and cluster genaralization values
        row :-  The record need to find the distance with cluster  type :- Pandas Series
        categorical_col :- Catergorical generalization values of the clusters   type :- Pandas DataFrame
        '''
        return (len(gv.GV['CAT_COL'])-np.where(
            row[gv.GV['CAT_COL']]==categorical_col,1,0).sum(axis=1))*10

    @staticmethod
    def numerical_dataloss(cluster,cluster_list,range_value,factor=20):
        '''
        This function used to find the numerical dataloss in a situation of two clusters join
        cluster :- Numerical values in the cluster type:-Pandas DataFrameGroupBy
        cluster_list :- Numerical values in the groups of clusters  type:-Pandas DataFrameGroupBy
        range_value :- Ranges for numerical values type :- Pandas DataFrame
        factor :- factor need to multiply. Recommended 20. type:- Integer
        '''
        return np.sum(((abs(cluster_list.min() - cluster.min()))+abs(cluster_list.max() - cluster.max()))/range_value*factor,axis=1)

    @staticmethod
    def categorical_dataloss(cluster,cluster_list,factor=20):
        '''
        This method is used to compare unique catergorical values in two clusters.
        cluster :- Categorical values in the cluster. type :- numpy array
        cluster_list :- Catergorical values in cluster values. type :- Pandas numpy array
        factor :- Factor to multiply the dataloss. Recomonded 20. Type Integer
        '''
        cluster = cluster.reshape((1,cluster.shape[0]))
        cluster_list = cluster_list.reshape((1,cluster_list.shape[0]))
        intersection_list = np.intersect1d(cluster[0],cluster_list[0])
        return abs(cluster[0].size+cluster_list[0].size-2*intersection_list.size)