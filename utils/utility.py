import pandas as pd

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

def get_full_span(df, categorical):
    for name in df.columns:
        if name not in categorical:
            df[name] = pd.to_numeric(df[name])

    return get_spans(df, categorical, df.index)

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


def diversity(df, partition, column):
    return len(df[column][partition].unique())


def is_l_diverse(df, partition, sensitive_column, l):
    return diversity(df, partition, sensitive_column) >= l


"""
@PARAMS
df - pandas dataframe
feature_column - list of column names along which to partitions the dataset
scale - column spans
is_valid - function to check the validity of a partition
"""


def partition_dataset(df, k, l, categorical, feature_columns, sensitive_column, scale):

    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns],
                          categorical, partition, scale)
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            lp, rp = split(df, categorical, partition, column)
            if l is None:
                if not is_k_anonymous(lp, k) or not is_k_anonymous(rp, k):
                    continue
            if l is not None:
                if not is_k_anonymous(lp, k) or not is_k_anonymous(rp, k) or not is_l_diverse(df, lp, sensitive_column, l) or not is_l_diverse(df, rp, sensitive_column, l):
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


def anonymizer(df, partitions, feature_columns, sensitive_column, categorical, max_partitions=None):
    aggregations = {}

    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
    rows = []

    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished {} partitions.".format(i))
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].assign(
            _common88column_=1).groupby('_common88column_').agg(aggregations, squeeze=False)
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
    return pdfn



