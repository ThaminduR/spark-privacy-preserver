import pandas as pd 

class Preserver:
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

    