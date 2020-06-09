import pandas as pd
import numpy as np
import time
import datetime
import random
from . import gv
from kmodes.kmodes import KModes
# from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
from .clustering_utils.input_validate import InputValidator
# import clustering_utils.input_validate.InputValidator

from .clustering_utils.data_loss import Dataloss
from .clustering_utils.clustering import Clustering
from .clustering_utils.cluster_init import ClusterInit
from .clustering_utils.kmodes import Kmodehelpers
from .clustering_utils.distance_calculation import Calculator


class Kanonymizer(object):
    def __init__(self, df, QI_attr, Sensitive_attr, cat_indecies, verbose=1, max_iter=10, anonimize_ratio=1, max_cluster_distance=20):
        '''

        Attributes:-
          df :- Pandas Dataframe that will used to anonymize. Type :- Pandas DataFrame
          QI_attr :- Column names of Quasi Identifiers. Type :- list
          Sensitive_attr :- Column names of Sensitive Columns
          verbose : Log details(1) or not(0).Default value is 1. Type : Boolean
          max_iter :- The maximum iteration number of joining clusters. Default value is 5 .Type :- Integer
          max_cluster_distance :- The maximum value in cluster distance. Default value is 20. Type :- Interger
        '''
        self.nan_replacement_int = 0
        self.nan_replacement_str = ''
        InputValidator.validate_input(df, QI_attr, Sensitive_attr, cat_indecies, verbose, max_iter,
                                      anonimize_ratio, max_cluster_distance, self.nan_replacement_int, self.nan_replacement_str)
        self.df = df
        self.df_copy = df.copy()
        self.df_second_copy = df.copy()
        self.QI_attr = QI_attr
        self.Sensitive_attr = Sensitive_attr
        self.n_clusters = 0
        self.verbose = verbose
        self.centroids = None
        self.less_centroids = None
        self.k_centroids = None
        self.k = 0
        self.max_iter = max_iter
        self.anonimize_ratio = anonimize_ratio
        self.max_cluster_distance = max_cluster_distance
        self.cluster_distances = None
        self.factor = 20

    def anonymize(self, k=10, mode='', center_type='fbcg', return_mode='Not_equal', iter=1):
        '''
        This method is used to anonymize the Dataset
        Parameters :- 
            k = Number of rows that cannot be distinguished from each other
            mode = if this is 'kmode', clustering will happen using KMODE clustering. Else it will happen in using Pandas Dataframe functions.
            center_type = Defines the method to choose cluster centers.
                  If method is not equal to kmode. Three values are possible.
                      1. 'fcbg' = This method return cluster centroids weight on the probability of row's column values appear in dataframe. Default Value.
                      2. 'rsc'  = This method will choose centroids weight according to the column that has most number of unique values.
                      3. 'random = Return cluster centroids in randomly.
            return_mode = If this value equal to 'equal' ; K anonymization will done with equal member clusters. Default value is 'Not_Equal'

        return :- 
            Anonymize dataset. Type :- Pandas DataFrame

        '''
        if (k <= 0):
            k = 10
        gv.k_global(k)
        self.k = int(k)
        self.n_clusters = len(self.df)//k
        self.c_centroids = np.zeros(len(self.df))
        if(self.verbose):
            print('K :' + str(k))
            print('Mode :' + str(mode))

        data = self.df.copy()
        unique_rows = data[gv.GV['QI']].drop_duplicates()
        mode = mode.lower()
        center_type = center_type.lower()
        return_mode = return_mode.lower()

        if(mode != 'komde'):
            if(self.verbose):
                print("Initializing centroidds")
            if(center_type == 'rsc'):
                self.centroids = self._random_sample_centroids(unique_rows)
            elif(center_type == 'random'):
                self.centroids = self._select_centroids_using_weighted_column(
                    unique_rows)
            else:
                self.centroids = self._find_best_cluster_gens(unique_rows)
            self.centroids.reset_index(drop=True, inplace=True)

        while(iter != 0):
            global row_number
            row_number = 0
            iter = iter-1
            if(mode != 'kmode'):
                if(self.verbose):
                    print('Clustering...')
                self.df['cluster_number'] = self.df.apply(
                    lambda row: self._clustering1(row), axis=1)
                if(return_mode == "equal"):
                    self._adjust_big_clusters1()
                    self._adjust_small_clusters1()
            else:
                if(center_type not in ['hung,cao']):
                    center_type = 'random'
                self.df = self._komode_clustering(
                    catergorical_indexes=gv.GV['CAT_INDEXES'], type_=center_type, n_init_=self.max_iter, verbose_=self.verbose)

            self._make_anonymize()
            self.anon_k_clusters()
            self.df[gv.GV['QI']] = self.df_second_copy[gv.GV['QI']]
            self.file_write()
            return self.df[gv.GV['QI']+gv.GV['SA']].applymap(str)

    def data_loss(self):
        """
        return complete data_loss
        input is the anonymized dataframe
        ouput is number between 0 and 1
        """
        return Dataloss.complete_data_loss(self.df, self.factor)

    # def _validate_input(self,df,QI_attr,Sensitive_attr,verbose,max_iter,anonimize_ratio,max_cluster_distance):
    #     """
    #     check input validity of the user inputs
    #     """
    #     InputValidator.validate_input(selfdf,QI_attr,Sensitive_attr,verbose,max_iter,anonimize_ratio,max_cluster_distance)
    #     return True

    def _level_cluster(self, cluster_num):
        """
        cluster_num is the number assigned to 
        """
        print(cluster_num)
        Clustering.level_cluster(self.df, cluster_num)

    def _adjust_big_clusters1(self):
        """
        Comment
        """
        Clustering.adjust_big_clusters1(self.df)

    def _adjust_small_clusters1(self):
        """
        Comment
        """
        num_of_iter = 0
        while(not(self.df['cluster_number'] != -1).all()):
            num_of_iter += 1
            best_clusters = (self.df.loc[self.df['cluster_number'] == -1]
                             ).apply(lambda row: self._clustering2(row), axis=1)
            self.df.at[best_clusters.index,
                       'cluster_number'] = best_clusters.values
            self._adjust_big_clusters1()
            if(num_of_iter >= self.max_iter):
                self.df = self.df.loc[self.df['cluster_number'] != -1]
                break

    def _clustering1(self, row):
        """
        Comment
        """
        global row_number
        row_number += 1
        best_cluster = Clustering.find_best_cluster(
            self.df, row, self.centroids)
        return best_cluster

    def _clustering2(self, row):
        """
        Comment
        """
        temp = self.df['cluster_number'].value_counts()
        small_clusters = temp.loc[temp < gv.k].index
        if (-1 in small_clusters):
            small_clusters.drop(-1)
        best_cluster = Clustering.find_best_cluster(
            self.df, row, self.centroids.iloc[small_clusters])
        return best_cluster

    def _komode_clustering(self, catergorical_indexes=[], type_='random', n_init_=3, verbose_=0):
        '''
        This method is used to clustering based on KMODE clustering.

        dataset : The pandas dataframe want to cluster. Type :- Pandas DataFrame
        catergorical_indexes = index of columns with catergorical type data. Type List 
        type_ = Method to clustering - Hung or Cao or random. Type String
        n_clusters_ = number of clusters. Type :- Integer 
        n_init_ = number of iterations to compare. Type :- Integer 
        verbose : Log details(1) or not(0). Type : Boolean
        '''
        km = KModes(self.n_clusters, init=type_,
                    n_init=n_init_, verbose=verbose_)
        y = km.fit_predict(self.df[gv.GV['QI']],
                           categorical=catergorical_indexes)
        columns = pd.DataFrame(km.cluster_centroids_, columns=gv.GV['QI'])

        columns[gv.GV['NUM_COL']] = columns[gv.GV['NUM_COL']
                                            ].applymap(lambda x: np.float(x))
        self.df[gv.GV['NUM_COL']] = self.df[gv.GV['NUM_COL']
                                            ].applymap(lambda x: np.float(x))
        columns[gv.GV['CAT_COL']] = columns[gv.GV['CAT_COL']
                                            ].applymap(lambda x: np.str(x))
        self.df[gv.GV['CAT_COL']] = self.df[gv.GV['CAT_COL']
                                            ].applymap(lambda x: np.str(x))
        self.df['cluster_number'] = list(km.labels_)
        non_zero_member_cluster_indices = self.df.groupby('cluster_number').filter(
            lambda grp: len(grp) != 0)['cluster_number'].unique()
        columns = columns.loc[non_zero_member_cluster_indices]
        columns = columns.reset_index()
        index_series = pd.Series(
            columns.index, index=non_zero_member_cluster_indices)
        self.c_centroids = columns
        self.df['cluster_number'] = self.df.apply(
            lambda row: index_series.loc[row['cluster_number']], axis=1)
        return self.df

    def _find_best_cluster_gens(self, dataframe):
        '''
        This method return cluster centroids weight on the probability of row's column values appear in dataframe.
        dataframe :- Unique rows in the dataframe. Type :- Pandas Dataframe
        '''

        return ClusterInit.find_best_cluster_gens(self.n_clusters, dataframe)

    def _select_centroids_using_weighted_column(self, dataframe):
        '''
        This method will choose centroids weight according to the column that has most number of unique values.
        dataframe :- Unique rows in the dataframe. Type :- Pandas Dataframe
        '''
        return ClusterInit.select_centroids_using_weighted_column(self.n_clusters, dataframe)

    def _random_sample_centroids(self, unique_rows):
        '''
        Return cluster centroids in randomly.
        dataframe :- Unique rows in the dataframe. Type :- Pandas Dataframe
        '''
        return ClusterInit.random_sample_centroids(self.n_clusters, unique_rows)

    def _make_anonymize(self, method='dataloss'):
        '''
        This method is used to generalize the dataframe after clustering
        method :- If this is dataloss, Distance of two clusters is measured by the dataloss of joining, else Distance between cluster centroids. Type :- String
        '''
        result = self._mark_clusters(method='dataloss')
        if(result == 1):
            return 1
        else:
            self.mark_less_clusters_to_kclusters()

    def _mark_clusters(self, method='dataloss'):
        '''
        This method is used join less member clusters to nearest cluster.
        method :- If this is dataloss, Distance of two clusters is measured by the dataloss of joining, else Distance between cluster centroids. Type :- String
        '''
        self.df_second_copy = self.df.copy()
        if(method == 'dataloss'):
            self.cluster_distances = self._cluster_data_loss()
        else:
            self.cluster_distances = self.less_centroids.apply(
                lambda row: self.get_distance_centers(row), axis=1)
        iteration_num = 0
        while(True):
            less_groups = self.df_second_copy.groupby('cluster_number').filter(
                lambda x: len(x) < self.k).groupby('cluster_number')
            if(less_groups.ngroups == 0):
                return 1
            elif(iteration_num >= self.max_iter):
                return 0
            else:
                # if(_DEBUG):                                                                                       ####################################################
                # self.mark_less_n_kcentroids()
                self.mark_less_clusters_to_close_clusters(self)
                iteration_num += 1

    def mark_less_n_kcentroids(self, dataframe='second'):
        '''
        This method is used to mark cluster centroids which has less than k number of members.
        dataframe :- The dataframe is used to couunt the cluster members. Type Pandas Dataframe
        '''
        if(dataframe == 'second'):
            dataframe = self.df_second_copy
        else:
            dataframe = self.df
        temp = dataframe['cluster_number'].value_counts()
        self.less_centroids = self.centroids.loc[temp.loc[temp <
                                                          self.k*self.anonimize_ratio].index]
        self.k_centroids = self.centroids.loc[temp.loc[temp >=
                                                       self.k*self.anonimize_ratio].index]

    def get_distance_centers(self, cluster_):
        '''
        This method is used to find the distances between cluster generalization values.
        cluster_ :- The cluster generalization value that need to find the distance. Type :- Pandas Series
        '''
        categorical_col = self.centroids[gv.GV['CAT_COL']]
        numerical_col = self.centroids[gv.GV['NUM_COL']]
        ranges = numerical_col.max() - numerical_col.min()
        return np.sum(cal_num_col_dist(cluster_[gv.GV['NUM_COL']], numerical_col, ranges, 20), axis=1) + cal_cat_col_dist3(cluster_, categorical_col)

    def _cluster_data_loss(self, apply_for='less_clusters', initialize=True):
        '''
        This method is used to find the dataloss of joining two clusters.
        This function return the dataloss among each and every clusters
        apply_for :- If this is 'less_clusters', Dataloss will find only for clusters that have less members than k. Type :- String
        initialize :- This parameter define Is it essential to initialize k less clusters or not. Type :- Boolean
        '''
        categorical_dataloss = np.vectorize(
            Calculator.categorical_dataloss, excluded="cluster_list")
        if(initialize):
            self.mark_less_n_kcentroids()
        self.less_centroids.sort_index(inplace=True)
        center_groups = self.df.groupby('cluster_number')
        center_num = center_groups[gv.GV['NUM_COL']]
        center_cat = center_groups[gv.GV['CAT_COL']]
        groups = center_groups.apply(lambda x: np.unique(
            np.concatenate(x[gv.GV['CAT_COL']].values).astype(str)))
        groups = np.array(groups)
        groups = groups.reshape((groups.shape[0], 1))
        ranges = center_num.max() - center_num.min() + gv.GV['RANGE_FIX']
        if(apply_for == 'less_clusters'):
            less_groups = self.df.groupby('cluster_number').filter(
                lambda x: len(x) < self.k).groupby('cluster_number')
            if(less_groups.ngroups == 0):
                return None
            less_lists = less_groups.apply(lambda x: np.unique(
                np.concatenate(x[gv.GV['CAT_COL']].values).astype(str)))
            less_lists = np.array(less_lists)
            less_lists = less_lists.reshape((less_lists.shape[0], 1))
            cat_distances = np.apply_along_axis(
                categorical_dataloss, 1, less_lists, groups)
            num_distance = less_groups.apply(lambda row: Calculator.numerical_dataloss(
                row[gv.GV['NUM_COL']], center_num, ranges))
            cat_frame_indices = self.less_centroids.index
        else:
            cat_distances = np.apply_along_axis(
                categorical_dataloss, 1, groups, groups)
            num_distance = center_groups.apply(lambda row: numerical_dataloss(
                row[gv.GV['NUM_COL']], center_num, ranges))
            cat_frame_indices = self.centroids.index
        shape = cat_distances.shape
        cat_distances = cat_distances.reshape(shape[0], shape[1]*shape[2])
        cat_frame = pd.DataFrame(cat_distances, index=cat_frame_indices)
        return cat_frame.add(num_distance, fill_value=gv.GV['QI_LEN']*self.max_cluster_distance)

    def mark_less_clusters_to_close_clusters(self, method='dataloss'):
        '''
        This method is used join less member clusters to nearest cluster.
        method :- If this is dataloss, Distance of two clusters is measured by the dataloss of joining, else Distance between cluster centroids. Type :- String
        '''
        self.mark_less_n_kcentroids()
        try:
            n_close_centroids = np.argsort(
                self.cluster_distances, axis=1).iloc[:, 1]
        except IndexError:
            return 1
        less_groups = self.df.groupby(
            'cluster_number').filter(lambda x: len(x) < self.k)
        groups = less_groups.groupby('cluster_number')
        less_cluster_indices = less_groups.index

        self.df_second_copy.at[less_cluster_indices, 'cluster_number'] = groups.apply(
            lambda grp: Kmodehelpers.edit_cluster(grp, n_close_centroids))

    def mark_less_clusters_to_kclusters(self, method='dataloss'):
        '''
        This method is used join less member clusters to nearest k or more member cluster.
        method :- If this is dataloss, Distance of two clusters is measured by the dataloss of joining, else Distance between cluster centroids. Type :- String
        '''
        self.mark_less_n_kcentroids()
        k_indices = self.df_second_copy.groupby('cluster_number').filter(
            lambda x: len(x) >= self.k)['cluster_number'].unique()
        cluster_distances = self.cluster_distances
        cluster_distances = cluster_distances[k_indices]

        try:
            if(k_indices.size == 1):
                n_close_centroids = np.argsort(
                    cluster_distances, axis=1).iloc[:, 0]
            else:
                n_close_centroids = np.argsort(
                    cluster_distances, axis=1).iloc[:, 1]
            cols = cluster_distances.columns
            n_close_centroids = n_close_centroids.apply(lambda row: cols[row])

        except IndexError:
            return 1
        less_groups = self.df_second_copy.groupby(
            'cluster_number').filter(lambda x: len(x) < self.k)
        groups = less_groups.groupby('cluster_number')
        less_cluster_indices = less_groups.index
        self.df_second_copy.at[less_cluster_indices, 'cluster_number'] = groups.apply(
            lambda grp: Kmodehelpers.edit_cluster(grp, n_close_centroids))

    def file_write(self, file_name='output.csv', sep_=',', encoding_='utf-8'):
        self.df[gv.GV['QI']+gv.GV['SA']
                ].to_csv(file_name, sep=sep_, encoding=encoding_)

    def anon_k_clusters(self):
        '''
        This method is used to generalize clusters.
        '''
        groups = self.df_second_copy.groupby('cluster_number')
        num_vals = groups[gv.GV['NUM_COL']].apply(
            Kmodehelpers.numeric_range).applymap(str)
        cat_vals = groups[gv.GV['CAT_COL']].apply(
            lambda row: row.apply(Kmodehelpers.catergorical_range))
        if(gv.GV['NUM_COL'] != []):
            anom_vals = num_vals.join(cat_vals)[gv.GV['QI']]
        else:
            anom_vals = cat_vals.join(num_vals)[gv.GV['QI']]
        self.df_second_copy[gv.GV['QI']] = self.df_second_copy.apply(
            lambda row: anom_vals.loc[row['cluster_number']], axis=1)

    def set_nan_replacement_int(replacement):
        self.nan_replacement_int = replacement

    def set_nan_replacement_str(replacement):
        self.nan_replacement_str = replacement


class LDiversityAnonymizer():
    def __init__(self, df, quasi_identifiers, sensitive_attributes, write_to_file=False, verbose=1):
        InputValidator.L_Diverse_Validate(
            df, quasi_identifiers, sensitive_attributes)
        self.df = df[quasi_identifiers + sensitive_attributes]
        self.sensitive_attributes = sensitive_attributes
        self.quasi_identifiers = quasi_identifiers
        self.verbose = verbose

########### Methods to write pandas dataframe to a file  #########################

    def file_write(self, file_name='output.csv', sep_=',', encoding_='utf-8'):
        self.df[quasi_identifiers +
                sensitive_attributes].to_csv(file_name, sep=sep_, encoding=encoding_)

########## Method to perform L Diversity #########################################

    def make_anonymize(self):
        l_diverse_rows = self.df.groupby(self.quasi_identifiers).filter(
            lambda group: self.count_sensitive(group))
        self.df = l_diverse_rows[self.quasi_identifiers +
                                 self.sensitive_attributes]

    def count_sensitive(self, grp):
        accept = True
        for column in self.sensitive_attributes:
            accept = accept and len(grp[column].unique()) >= self.l
        return accept

######### Public method  ########################################################

    def anonymize(self, l=2):
        if(l < 2):
            l = 2
        self.l = int(l)
        self.make_anonymize()
        return self.df


class TClosenessAnonymizer():

    def __init__(self, df, quasi_identifiers, sensitive_attributes, write_to_file=False, verbose=1):
        InputValidator.L_Diverse_Validate(
            df, quasi_identifiers, sensitive_attributes)
        self.df = df[quasi_identifiers + sensitive_attributes]
        self.sensitive_attributes = sensitive_attributes
        self.quasi_identifiers = quasi_identifiers
        self.verbose = verbose
        self.thresholds = None

########### Methods to write pandas dataframe to a file  #########################

    def file_write(self, file_name='output.csv', sep_=',', encoding_='utf-8'):
        self.df[quasi_identifiers +
                sensitive_attributes].to_csv(file_name, sep=sep_, encoding=encoding_)

########## Method to perform T Closeness #########################################

    def make_anonymize(self):
        self.define_thresholds()
        t_closeness_rows = self.df.groupby(self.quasi_identifiers).filter(
            lambda group: self.check_thresholds(group))
        self.df = t_closeness_rows[self.quasi_identifiers +
                                   self.sensitive_attributes]

    def define_thresholds(self):
        thresholds = {column: self.df[column].value_counts(
        )/len(self.df) for column in self.sensitive_attributes}
        self.thresholds = thresholds

    def check_thresholds(self, grp):
        length_cluster = len(grp)
        accept = True
        for column in self.sensitive_attributes:
            grp_thresholds = grp[column].value_counts()/length_cluster
        for element in grp_thresholds.keys():
            accept = accept and (
                grp_thresholds[element]) + self.t >= self.thresholds[column][element]
        return accept

######### Public method  ########################################################

    def anonymize(self, t=0.2):
        if(t >= 1):
            self.t = 0.2
        else:
            self.t = t
        self.make_anonymize()

        return self.df
