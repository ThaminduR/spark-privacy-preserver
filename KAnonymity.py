import pandas as pd 

class KAnonymity:

    def __init__(self,df,categorical):
        self.df = df.select("*").toPandas()
        self.categorical = categorical

    """
    @PARAMS
    df - spark.sql dataframe
    partition - parition for whic to calculate the spans
    scale: if given, the spans of each column will be divided 
           by the scale for that column
    """
    def __get_spans(self,df, partition, scale=None):
        
        columns = list(df.columns)
        categorical = self.categorical
        spans = {}
        for column in df.columns:
            if column in categorical:
                span = len(df[column][partition].unique())
            else:
                span = df[column][partition].max()-df[column][partition].min()
            if scale is not None:
                span = span/scale[column]
            spans[column] = span
        return spans


    """
    @PARAMS
    df - spark.sql dataframe
    partition - parition for whic to calculate the spans
    column: column to split
    """
    def __split(self,df, partition, column):

        categorical = self.categorical
        dfp = df[column][partition]
        if column in categorical:
            values = dfp.unique()
            lv = set(values[:len(values)//2])
            rv = set(values[len(values)//2:])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
        else:        
            median = dfp.median()
            dfl = dfp.index[dfp < median]
            dfr = dfp.index[dfp >= median]
            return (dfl, dfr)


    def __is_k_anonymous(self,df, partition, sensitive_column, k=3):
    
        if len(partition) < k:
            return False
        return True


    """
        :param               df: The dataframe to be partitioned.
        :param  feature_columns: A list of column names along which to partition the dataset.
        :param sensitive_column: The name of the sensitive column (to be passed on to the `is_valid` function)
        :param            scale: The column spans as generated before.
        :param         is_valid: A function that takes a dataframe and a partition and returns True if the partition is valid.
        :returns               : A list of valid partitions that cover the entire dataframe.
        """

    """
    @PARAMS
    df - spark.sql dataframe
    feature_column - list of column names along which to partitions the dataset
    scale - column spans
    is_valid - function to check the validity of a partition
    """
    def __partition_dataset(self,df, feature_columns, sensitive_column, scale, is_valid):
        
        finished_partitions = []
        partitions = [df.index]
        while partitions:
            partition = partitions.pop(0)
            spans = self.__get_spans(df[feature_columns], partition, scale)
            for column, span in sorted(spans.items(), key=lambda x:-x[1]):
                lp, rp = self.__split(df, partition, column)
                if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        return finished_partitions


    def __build_indexes(self,df):
        indexes = {}
        categorical = self.categorical
        for column in categorical:
            values = sorted(df[column].unique())
            indexes[column] = { x : y for x, y in zip(values, range(len(values)))}
        return indexes



