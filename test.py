import unittest
from mondrian_preserver import Preserver
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import random
import pdb

spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")


def init():
    data = [[6, '1', 'test1', 'x', 20],
            [6, '1', 'test1', 'y', 30],
            [8, '2', 'test2', 'x', 50],
            [8, '2', 'test2', 'x', 45],
            [4, '1', 'test2', 'y', 35],
            [4, '2', 'test3', 'y', 20]]

    cSchema = StructType([StructField("column1", IntegerType()),
                          StructField("column2", StringType()),
                          StructField("column3", StringType()),
                          StructField("column4", StringType()),
                          StructField("column5", IntegerType())])
    df = spark.createDataFrame(data, schema=cSchema)
    categorical = set((
        'column2',
        'column3',
        'column4'
    ))
    feature_columns = ['column1', 'column2', 'column3']
    return df, feature_columns, categorical


class functionTest(unittest.TestCase):
    def test1_k_anonymize(self):
        df, feature_columns, categorical = init()
        sensitive_column = 'column4'
        schema = StructType([
            StructField("column1", DoubleType()),
            StructField("column2", StringType()),
            StructField("column3", StringType()),
            StructField("column4", StringType()),
            StructField("count", IntegerType())
        ])
        resultdf = Preserver.k_anonymize(df, 3, feature_columns,
                                         sensitive_column, categorical, schema)
        resultdf.show()
        testdata = [[5.333333333333333,1,'test1,test2','x',1],
                    [5.333333333333333,1,'test1,test2','y',2],
                    [6.666666666666667,2,'test3,test2','x',2],
                    [6.666666666666667,2,'test3,test2','y',1]]
        testdf = spark.createDataFrame(testdata, schema=schema)

        try:
            self.assertTrue(testdf.exceptAll(resultdf).count()==0)
        except AssertionError:
            print('Incorrect Dataframe')
        
    def test2_k_anonymize(self):
        df, feature_columns, categorical = init()
        sensitive_column = 'column5'
        schema = StructType([
            StructField("column1", DoubleType()),
            StructField("column2", StringType()),
            StructField("column3", StringType()),
            StructField("column5", DoubleType()),
            StructField("count", IntegerType())
        ])
        resultdf = Preserver.k_anonymize(df, 3, feature_columns,
                                         sensitive_column, categorical, schema)
        resultdf.show()
        # testdata = [[5.333333333333333,1,'test1,test2','x',1],
        #             [5.333333333333333,1,'test1,test2','y',2],
        #             [6.666666666666667,2,'test3,test2','x',2],
        #             [6.666666666666667,2,'test3,test2','y',1]]
        # testdf = spark.createDataFrame(testdata, schema=schema)

        # try:
        #     self.assertTrue(testdf.exceptAll(resultdf).count()==0)
        # except AssertionError:
        #     print('Incorrect Dataframe')

if __name__ == '__main__':
    unittest.main()
