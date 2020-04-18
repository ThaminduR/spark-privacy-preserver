import json
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
from functools import wraps
from pyspark.sql.types import MapType, StructType, ArrayType, StructField
from pyspark.sql.functions import to_json, from_json


def is_complex_dtype(dtype):
    """Check if dtype is a complex type

    Args:
        dtype: Spark Datatype

    Returns:
        Bool: if dtype is complex
    """
    return isinstance(dtype, (MapType, StructType, ArrayType))


def complex_dtypes_to_json(df):
    """Converts all columns with complex dtypes to JSON

    Args:
        df: Spark dataframe

    Returns:
        tuple: Spark dataframe and dictionary of converted columns and their data types
    """
    conv_cols = dict()
    selects = list()
    for field in df.schema:
        if is_complex_dtype(field.dataType):
            conv_cols[field.name] = field.dataType
            selects.append(to_json(field.name).alias(field.name))
        else:
            selects.append(field.name)
    df = df.select(*selects)
    return df, conv_cols


def complex_dtypes_from_json(df, col_dtypes):
    """Converts JSON columns to complex types

    Args:
        df: Spark dataframe
        col_dtypes (dict): dictionary of columns names and their datatype

    Returns:
        Spark dataframe
    """
    selects = list()
    for column in df.columns:
        if column in col_dtypes.keys():
            schema = StructType([StructField('root', col_dtypes[column])])
            selects.append(from_json(column, schema).getItem(
                'root').alias(column))
        else:
            selects.append(column)
    return df.select(*selects)


def toPandas(df):
    """Same as df.toPandas() but converts complex types to JSON first

    Args:
        df: Spark dataframe

    Returns:
        Pandas dataframe
    """
    return complex_dtypes_to_json(df)[0].toPandas()


def cols_from_json(df, columns):
    """Converts Pandas dataframe colums from json

    Args:
        df (dataframe): Pandas DataFrame
        columns (iter): list of or iterator over column names

    Returns:
        dataframe: new dataframe with converted columns
    """
    for column in columns:
        df[column] = df[column].apply(json.loads)
    return df


def ct_val_to_json(value):
    """Convert a scalar complex type value to JSON

    Args:
        value: map or list complex value

    Returns:
        str: JSON string
    """
    return json.dumps({'root': value})


def cols_to_json(df, columns):
    """Converts Pandas dataframe columns to json and adds root handle

    Args:
        df (dataframe): Pandas DataFrame
        columns ([str]): list of column names

    Returns:
        dataframe: new dataframe with converted columns
    """
    for column in columns:
        df[column] = df[column].apply(ct_val_to_json)
    return df


class pandas_udf_ct(object):
    """Decorator for UDAFs with Spark >= 2.3 and complex types

    Args:
        returnType: the return type of the user-defined function. The value can be either a 
                    pyspark.sql.types.DataType object or a DDL-formatted type string.
        functionType: an enum value in pyspark.sql.functions.PandasUDFType. Default: SCALAR.

    Returns:
        Function with arguments `cols_in` and `cols_out` defining column names having complex 
        types that need to be transformed during input and output for GROUPED_MAP. In case of 
        SCALAR, we are dealing with a series and thus transformation is done if `cols_in` or 
        `cols_out` evaluates to `True`. 
        Calling this functions with these arguments returns the actual UDF.
    """

    def __init__(self, returnType=None, functionType=None):
        self.return_type = returnType
        self.function_type = functionType

    def __call__(self, func):
        @wraps(func)
        def converter(*, cols_in=None, cols_out=None):
            if cols_in is None:
                cols_in = list()
            if cols_out is None:
                cols_out = list()

            @pandas_udf(self.return_type, self.function_type)
            def udf_wrapper(values):
                if isinstance(values, pd.DataFrame):
                    values = cols_from_json(values, cols_in)
                elif isinstance(values, pd.Series) and cols_in:
                    values = values.apply(json.loads)
                res = func(values)
                if self.function_type == PandasUDFType.GROUPED_MAP:
                    if isinstance(res, pd.Series):
                        res = res.to_frame().T
                    res = cols_to_json(res, cols_out)
                elif cols_out and self.function_type == PandasUDFType.SCALAR:
                    res = res.apply(ct_val_to_json)
                elif (isinstance(res, (dict, list)) and
                      self.function_type == PandasUDFType.GROUPED_AGG):
                    res = ct_val_to_json(res)
                return res

            return udf_wrapper

        return converter

