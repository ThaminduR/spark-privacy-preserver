
from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
from utils.utility import *


def k_anonymizer(df, k, feature_columns, sensitive_column, categorical):

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, None, None,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymizer(df, partitions, feature_columns, sensitive_column, categorical)


def l_diversity_anonymizer(df, k, l, feature_columns, sensitive_column, categorical):

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, l, None,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymizer(df, partitions, feature_columns, sensitive_column, categorical)


def t_closeness_anonymizer(df, k, t, feature_columns, sensitive_column, categorical):
    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, None, t,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymizer(df, partitions, feature_columns, sensitive_column, categorical)


class Preserver:

    @staticmethod
    def k_anonymize(df, k, feature_columns, sensitive_column, categorical, schema):

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymize(pdf):
            a_df = k_anonymizer(pdf, k, feature_columns,
                                sensitive_column, categorical)
            return a_df

        return df.groupby().apply(anonymize)

    @staticmethod
    def l_diversity(df, k, l, feature_columns, sensitive_column, categorical, schema):

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymize(pdf):
            a_df = l_diversity_anonymizer(pdf, k, l, feature_columns,
                                          sensitive_column, categorical)
            return a_df

        return df.groupby().apply(anonymize)

    @staticmethod
    def t_closeness(df, k, t, feature_columns, sensitive_column, categorical, schema):

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymize(pdf):
            a_df = t_closeness_anonymizer(pdf, k, t, feature_columns,
                                          sensitive_column, categorical)
            return a_df

        return df.groupby().apply(anonymize)

    @staticmethod
    def anonymize_user(df, k, user, usercolumn_name, sensitive_column, categorical, schema, random=False):

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymize(pdf):
            a_df = user_anonymizer(
                pdf, k, user, usercolumn_name, sensitive_column, categorical, random)
            return a_df

        return df.groupby().apply(anonymize)
