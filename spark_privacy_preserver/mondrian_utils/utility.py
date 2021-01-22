import hashlib
import pandas as pd

# """Custom Error class"""


class AnonymizeError(Exception):
    def __init__(self, message):
        self.message = message


# """
# @PARAMS - get_spans()
# df - pandas dataframe
# partition - parition for whic to calculate the spans
# scale: if given, the spans of each column will be divided
#         by the scale for that column
# """


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


# """
# @PARAMS - split()
# df - pandas dataframe
# partition - parition for whic to calculate the spans
# column: column to split
# """


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


def l_diversity(df, partition, column):
    return len(df[column][partition].unique())


def is_l_diverse(df, partition, sensitive_column, l):
    return l_diversity(df, partition, sensitive_column) >= l


# """
# @PARAMS - t_closeness()
# global_freqs: The global frequencies of the sensitive attribute values

# """


def t_closeness(df, partition, column, global_freqs):
    total_count = float(len(partition))
    d_max = None
    group_counts = df.loc[partition].groupby(column)[column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count/total_count
        d = abs(p-global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max


# """
# @PARAMS - is_t_close()
# global_freqs: The global frequencies of the sensitive attribute values
# p: The maximum aloowed distance
# """


def is_t_close(df, partition, categorical, sensitive_column, global_freqs, p):

    if not sensitive_column in categorical:
        raise ValueError("T closeness is only for categorical values")
    result = t_closeness(df, partition, sensitive_column, global_freqs) <= p
    if(result):
        return result
    else:
        print("No T closseness")


def get_global_freq(df, sensitive_column):
    global_freqs = {}
    total_count = float(len(df))
    group_counts = df.groupby(sensitive_column)[sensitive_column].agg('count')

    for value, count in group_counts.to_dict().items():
        p = count/total_count
        global_freqs[value] = p
    return global_freqs


# @PARAMS - partition_dataset()
# df - pandas dataframe
# feature_column - list of column names along which to partitions the dataset
# scale - column spans


def partition_dataset(df, k, l, t, categorical, feature_columns, sensitive_column, scale):

    finished_partitions = []
    global_freqs = {}
    if t is not None:
        global_freqs = get_global_freq(df, sensitive_column)

    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns],
                          categorical, partition, scale)
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            lp, rp = split(df, categorical, partition, column)
            if l is not None:
                if not is_k_anonymous(lp, k) or not is_k_anonymous(rp, k) or not is_l_diverse(df, lp, sensitive_column, l) or not is_l_diverse(df, rp, sensitive_column, l):
                    continue
            if l is None:
                if t is None:
                    if not is_k_anonymous(lp, k) or not is_k_anonymous(rp, k):
                        continue
                if t is not None:
                    if not is_k_anonymous(lp, k) or not is_k_anonymous(rp, k) or not is_t_close(df, lp, categorical, sensitive_column, global_freqs, t) or not is_t_close(df, rp, categorical, sensitive_column, global_freqs, t):
                        continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions


def agg_categorical_column(series):
    return ','.join(set(series))


def agg_numerical_column(series):
    minimum = series.min()
    maximum = series.max()
    if(maximum == minimum):
        string = str(maximum)
    else:
        string = ''
        maxm = str(maximum)
        minm = str(minimum)

        if(len(minm) == 1):
            if(minimum >= 5):
                string = '5-'
            else:
                string = '0-'
        else:
            if (minm[-1]=='0'):
            string = minm +"-"
            else:
                min_start = minm[:-1]
                if(minimum >= int(min_start+'5')):
                    string = min_start+'5-'
                else:
                    string = min_start+'0-'

        if(len(maxm) == 1):
            if(maximum >= 5):
                string += "10"
            else:
                string += '5'
        else:
            if(maxm[-1]=='0'):
            string += maxm
            else:
                max_start = maxm[:-1]
                if(maximum > int(max_start+'5')):
                    string += str(int(max_start+'0') + 10)
                else:
                    string += max_start+'5'

    return string

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
            print("Finished processing {} partitions.".format(i))
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


# """ --------------------------------------------------------------------------
    # Single User Anonymize
# """ --------------------------------------------------------------------------


def getIntersection(df, udf, user, threshold, columns, usercolumn_name):
    i = 0
    intersect_df = pd.DataFrame()
    for column in columns:
        i += 1
        if (i > threshold):
            break
        val = udf[column].value_counts().idxmax()
        tempdf = df.loc[(df[column] == val) & (df[usercolumn_name] != user)]
        if (intersect_df.empty):
            intersect_df = tempdf
        if(not tempdf.empty):
            intersect_df = tempdf.reset_index().merge(
                intersect_df, how='inner').set_index('index')

    return intersect_df


def commonDF(df, udf, user, requiredRows, columns, usercolumn_name, random):
    length = len(columns)
    for i in range(length):
        intersect_df = getIntersection(
            df, udf, user, length-i, columns, usercolumn_name)
        if(intersect_df.shape[0] >= requiredRows):
            break
        else:
            continue

    revcolumn = columns[::-1]

    for i in range(length):
        intersect_df = getIntersection(
            df, udf, user, length-i, revcolumn, usercolumn_name)
        if(intersect_df.shape[0] >= requiredRows):
            break
        else:
            continue

    if (intersect_df.shape[0] < requiredRows):
        for column in columns:
            intersect_df = getIntersection(
                df, udf, user, 1, [column], usercolumn_name)
            if(intersect_df.shape[0] >= requiredRows):
                break
    if ((intersect_df.shape[0] < requiredRows) & random):
        try:
            intersect_df = df.sample(requiredRows)
        except ValueError:
            raise(AnonymizeError("Data frame is not enough for anonymization"))
    return intersect_df


def anonymizeGivenUser(df, udf, user, usercolumn_name, columns, categorical):
    indexes = list(udf.index)
    for column in columns:
        if column not in categorical:
            udf[column] = pd.to_numeric(udf[column])
            df[column] = pd.to_numeric(df[column])
        valueList = udf[column].unique()
        if column in categorical:
            string = ','.join(valueList)
            df[column] = df[column].astype(str)
            df.loc[indexes, column] = string
        if column not in categorical:

            minimum = min(valueList)
            maximum = max(valueList)
            if(maximum == minimum):
                string = str(maximum)
            else:
                string = ''
                maxm = str(maximum)
                minm = str(minimum)

                if(len(minm) == 1):
                    min_start = minm[-1]
                    if(minimum >= 5):
                        string = '5-'
                    else:
                        string = '0-'
                else:
                    min_start = minm[-2]
                    if(minimum >= int(min_start+'5')):
                        string = min_start+'5-'
                    else:
                        string = min_start+'0-'

                if(len(maxm) == 1):
                    max_start = maxm[-1]
                    if(maximum >= 5):
                        string += "10"
                    else:
                        string += '5'
                else:
                    max_start = maxm[-2]
                    if(maximum >= int(max_start+'5')):
                        string += str(int(max_start+'0') + 10)
                    else:
                        string += max_start+'5'

                        min_start = minm[-2]
                        max_start = maxm[-2]

            df[column] = df[column].astype(str)
            df.loc[indexes, column] = string


def user_anonymizer(df, k, user, usercolumn_name, sensitive_column, categorical, random=False):

    if ((sensitive_column not in df.columns) or (usercolumn_name not in df.columns)):
        raise AnonymizeError("No Such Sensitive Column")

    df[usercolumn_name] = df[usercolumn_name].astype(str)

    userdf = df.loc[df[usercolumn_name] == str(user)]
    user = str(user)
    if(userdf.empty):
        raise AnonymizeError("No user found.")

    rowcount = userdf.shape[0]
    columns = userdf.columns.drop([usercolumn_name, sensitive_column])

    if (rowcount >= k):
        requiredRows = 1
    else:
        requiredRows = k - rowcount
    intersect_df = commonDF(
        df, userdf, user, requiredRows, columns, usercolumn_name, random)

    if((not intersect_df.empty) & (intersect_df.shape[0] >= requiredRows)):
        finaldf = pd.concat([userdf, intersect_df])
        anonymizeGivenUser(df, finaldf, user,
                           usercolumn_name, columns, categorical)
    else:
        raise(AnonymizeError("Can't K Anonymize the user for given K value"))
    return df


# """ --------------------------------------------------------------------------
    # Anonymize with all rows
# """ --------------------------------------------------------------------------
def agg_columns(df, partdf, indexes, columns, categorical):

    for column in columns:

        if column not in categorical:
            partdf[column] = pd.to_numeric(partdf[column])
        valueList = partdf[column].unique()

        if column in categorical:
            string = ','.join(valueList)
            df[column] = df[column].astype(str)
            df.loc[indexes, column] = string

        if column not in categorical:
            minimum = min(valueList)
            maximum = max(valueList)
            if(maximum == minimum):
                string = str(maximum)
            else:
                string = ''
                maxm = str(maximum)
                minm = str(minimum)
                if(len(minm) == 1):
                    if(minimum >= 5):
                        string = '5-'
                    else:
                        string = '0-'
                else:
                    if (minm[-1]=='0'):
                        string = minm +"-"
                    else:
                        min_start = minm[:-1]
                        if(minimum >= int(min_start+'5')):
                            string = min_start+'5-'
                        else:
                            string = min_start+'0-'

                if(len(maxm) == 1):
                    if(maximum >= 5):
                        string += "10"
                    else:
                        string += '5'
                else:
                    if(maxm[-1]=='0'):
                        string += maxm
                    else:
                        max_start = maxm[:-1]
                        if(maximum > int(max_start+'5')):
                            string += str(int(max_start+'0') + 10)
                        else:
                            string += max_start+'5'

            df[column] = df[column].astype(str)
            df.loc[indexes, column] = string


def anonymize_w_user(df, partitions, feature_columns, sensitive_column, categorical):

    if sensitive_column not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for fcolumn in feature_columns:
        if fcolumn not in df.columns:
            raise AnonymizeError("No Such Feature Column :"+fcolumn)

    full_spans = get_full_span(df, categorical)
    aggregations = {}
    df_copy = df.copy()
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column

    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished processing {} partitions.".format(i))

        partdf = df.loc[partition]
        agg_columns(df, partdf, partition, feature_columns, categorical)

    df = df.sort_values(feature_columns+[sensitive_column])
    return df
