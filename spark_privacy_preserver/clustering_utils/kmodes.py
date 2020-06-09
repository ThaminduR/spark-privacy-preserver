import numpy as np

class Kmodehelpers:

    @staticmethod
    def edit_cluster(group,distances):
        '''
        This method is used to edit the cluster number of a record.
        group :- Cluster group, type = Pandas DataFrameGroupBy
        distances :- Dataframe that include cluster mapping details
        '''
        c_no = group['cluster_number'].unique()[0]
        group['cluster_number'] = distances[c_no]
        return group

    @staticmethod
    def numeric_range(numeric_values):
        '''
        This method is used to generalize the data in a cluster.
        The given values in a cluster turn to the min-max range if they are not same.
        numeric_values :- Numeric values of the cluster. type = Pandas DataFrame
        '''
        numerical_min = numeric_values.min()
        numerical_max = numeric_values.max()
        return numerical_min.where(numerical_min.eq(numerical_max),numerical_min.apply(str) +"-"+numerical_max.apply(str))

    @staticmethod
    def catergorical_range(catergorical):
        '''
        This method is used to anonymize catergorical values in a cluster.
        Given categorical values turn to comma seperated generalization if they are not same.
        catergorical :- Catergorical values in a cluster. type = Pandas DataFrame
        '''
        catergorical = catergorical.apply(str)
        return ','.join(catergorical.unique())

    def remove_far_clusters(self):
        '''
        This method is used to remove the far away clusters according to users preference
        Class attribute anonimize_ratio used to measure the distance
        '''
        temp = self.df['cluster_number'].value_counts()
        very_small_clusters = temp.loc[temp <= self.k*self.anonimize_ratio].index
        not_small_cluster = self.df.apply(lambda row: row['cluster_number'] not in very_small_clusters,axis=1)
        self.centroids = self.centroids.iloc[temp.loc[temp > self.k*self.anonimize_ratio].index]
        self.df = self.df.loc[not_small_cluster]
        if(_DEBUG):
            print(self.df)
            print(self.centroids)
        return self.df.sort_values(by=['cluster_number']),self.centroids


    def adjust_kless_clusters(self,n = 2,method='dataloss'):
        '''
        This method is used to anonymize clusters that have less than k members.
        n :- Define the number of clusters that will join with a single less member cluster. Type :- Integer
        method :- If this is dataloss, Distance of two clusters is measured by the dataloss of joining, else Distance between cluster centroids. Type :- String
        '''
        self.mark_less_n_kcentroids()
        if(_DEBUG):
          print("marked_less clusters")
        self.df_copy = self.df.copy()
        self.df_copy['is_anonimized'] = 0
        if(method == 'dataloss'):
            centroid_distances = self.cluster_data_loss()
        else: 
            centroid_distances = self.less_centroids.apply(lambda row : self.get_distance_centers(row),axis=1)
        if(_DEBUG):
          print("Get cenr distance")
        n_close_centroids = np.argsort(centroid_distances,axis=1).iloc[:,:n]
        if(_DEBUG):
          print("After taking closing")
        results =  self.less_centroids.apply(lambda row: self.anonimize_k_less_clusters(row,n_close_centroids),axis=1)
        return results

    def anonimize_k_less_clusters(self,centroid,n_close_centroids):
        '''
        This method used to anonymize a single less member cluster
        centroid:- The centroid of the cluster. Type :- Pandas Series
        n_close_centroids :- This have the details about distances between cluster.(Distances measure by dataloss or centroid distances). Type :- Pandas DataFrame        
        '''
        if(_DEBUG):
          print("anonimize_k_less_clusters")
        close_cetroids = n_close_centroids.loc[centroid.name]
        if(_DEBUG):
          print("take_n_close")
        centroid_related_rows = self.df.loc[self.df.apply(lambda x:x['cluster_number'] in close_cetroids.values,axis=1)].sort_values(by=['cluster_number'])
        if(_DEBUG):
          print("centroid_related_rows")
        related_indexes = self.df_copy.apply(lambda x: x['cluster_number'] == centroid.name,axis=1)
        if(_DEBUG):
          print("related_indexes")
        numerical_min = centroid_related_rows[NUM_COL].min()
        numerical_max = centroid_related_rows[NUM_COL].max()
        centroid_related_rows[NUM_COL] = np.where(numerical_min == numerical_max,numerical_min,numerical_min.apply(str) +"-"+numerical_max.apply(str))
        centroid_related_rows[CAT_COL] = np.where(centroid_related_rows[CAT_COL].nunique() == 1,centroid_related_rows[CAT_COL],centroid_related_rows[CAT_COL].apply(anonimize_catergorical_value_in_cluster))
        centroid_related_rows = centroid_related_rows.loc[related_indexes]
        if(_DEBUG):
          print("centroid_related_rows")
        self.df_copy.at[related_indexes,'is_anonimized'] = 1
        self.df_copy.at[related_indexes,QI] = centroid_related_rows[QI]
        if(_DEBUG):
          print("above to finish k less")
        return centroid_related_rows


