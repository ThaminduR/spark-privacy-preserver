import pandas as pd


class KAnonymity:

    def __init__(self, df, categorical):
        #self.df = df.select("*").toPandas()
        self.df = df
        self.categorical = categorical
        for name in categorical:
            df[name] = df[name].astype('category')
    """
    @PARAMS
    df - spark.sql dataframe
    partition - parition for whic to calculate the spans
    scale: if given, the spans of each column will be divided 
           by the scale for that column
    """

    def __get_spans(self, df, partition, scale=None):

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

    def __split(self, df, partition, column):

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

    def __is_k_anonymous(self, df, partition, sensitive_column, k=3):

        if len(partition) < k:
            return False
        return True


    """
    @PARAMS
    df - spark.sql dataframe
    feature_column - list of column names along which to partitions the dataset
    scale - column spans
    is_valid - function to check the validity of a partition
    """

    def __partition_dataset(self, df, feature_columns, sensitive_column, scale, is_valid):

        finished_partitions = []
        partitions = [df.index]
        while partitions:
            partition = partitions.pop(0)
            spans = self.__get_spans(df[feature_columns], partition, scale)
            for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                lp, rp = self.__split(df, partition, column)
                if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        return finished_partitions


    def __agg_categorical_column(self, series):
        return [','.join(set(series))]


    def __agg_numerical_column(self, series):
        return [series.mean()]


    def build_anonymized_dataset(self, spark, feature_columns, sensitive_column, max_partitions=None):
        aggregations = {}
        df = self.df
        full_spans = self.__get_spans(df, df.index)
        partitions = self.__partition_dataset(
            df, feature_columns, sensitive_column, full_spans, self.__is_k_anonymous)
        categorical = self.categorical
        for column in feature_columns:
            if column in categorical:
                aggregations[column] = self.__agg_categorical_column
            else:
                aggregations[column] = self.__agg_numerical_column
        rows = []

        for i, partition in enumerate(partitions):
            if i % 100 == 1:
                print("Finished {} partitions...".format(i))
            if max_partitions is not None and i > max_partitions:
                break
            grouped_columns = df.loc[partition].agg(
                aggregations, squeeze=False)
            sensitive_counts = df.loc[partition].groupby(
                sensitive_column).agg({sensitive_column: 'count'})
            values = grouped_columns.iloc[0].to_dict()
            for sensitive_value, count in sensitive_counts[sensitive_column].items():
                if count == 0:
                    continue
                values.update({
                    sensitive_column: sensitive_value,
                    'count': count,

                })
                rows.append(values.copy())
        dfn = pd.DataFrame(rows)
        pdfn = dfn.sort_values(feature_columns+[sensitive_column])
        dfn = spark.createDataFrame(pdfn)
        return dfn
