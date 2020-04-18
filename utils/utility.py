
"""
@PARAMS
df - pandas dataframe
partition - parition for whic to calculate the spans
scale: if given, the spans of each column will be divided 
        by the scale for that column
"""

def get_spans(df, categorical, partition, scale=None):

    columns = list(df.columns)
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
df - pandas dataframe
partition - parition for whic to calculate the spans
column: column to split
"""

def split(df, categorical, partition, column):
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

def is_k_anonymous(partition, k):

    if len(partition) < k:
        return False
    return True

"""
@PARAMS
df - pandas dataframe
feature_column - list of column names along which to partitions the dataset
scale - column spans
is_valid - function to check the validity of a partition
"""

def partition_dataset(df, k, categorical, feature_columns, sensitive_column, scale, is_valid):

    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns],
                                    categorical, partition, scale)
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            lp, rp = split(df, categorical, partition, column)
            if not is_valid(lp, k) or not is_valid(rp, k):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions

def agg_categorical_column(series):
    return ','.join(set(series))

def agg_numerical_column(series):
    return series.mean()