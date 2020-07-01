from spark_privacy_preserver.differential_privacy import DPLib
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

import unittest
from random import random, randint, choice
import json
import sys
import contextlib
import io
import time


class DPLibTestCase(unittest.TestCase):

    def test_init(self):

        for case in epsilon_true:
            dp = DPLib(global_epsilon=case, global_delta=0.5)
            self.assertEqual(dp._DPLib__epsilon, case, msg='testing method: `init`, case: `epsilon_true` failed')
            del dp

        for case in epsilon_false:
            try:
                DPLib(global_epsilon=case)
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertIn(member=(str(exc_type), str(exc_msg)),
                              container=(("<class 'TypeError'>", 'Epsilon and delta must be numeric'),
                                         ("<class 'ValueError'>", 'Epsilon must be non-negative'),
                                         ("<class 'ValueError'>", 'Epsilon and Delta cannot both be zero')))
                continue

        for case in delta_true:
            dp = DPLib(global_epsilon=0.1, global_delta=case)
            self.assertEqual(dp._DPLib__delta, case, msg='testing method: `init`, case: `delta_true` failed')
            del dp

        for case in delta_false:
            try:
                DPLib(global_epsilon=0, global_delta=case)
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertIn(member=(str(exc_type), str(exc_msg)),
                              container=(("<class 'TypeError'>", 'Epsilon and delta must be numeric'),
                                         ("<class 'ValueError'>", 'Delta must be in range [0, 1]'),
                                         ("<class 'ValueError'>", 'Epsilon and Delta cannot both be zero')),
                              msg='testing method: `init`, case: `delta_false` failed')

                continue

        dp = DPLib(sdf=sdf_true)
        self.assertEqual(dp.sdf, sdf_true, msg='testing method: `init`, case: `sdf_true` failed')
        del dp

        for case in sdf_false:
            try:
                DPLib(sdf=case)
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertEqual(first=(str(exc_type), str(exc_msg)),
                                 second=("<class 'TypeError'>", 'Given sdf is not a Spark DataFrame'),
                                 msg='testing method: `init`, case: `sdf_false` failed')

                continue

    def test_set_global_epsilon_delta(self):

        for case in epsilon_true:
            dp = DPLib()
            dp.set_global_epsilon_delta(epsilon=case, delta=0.5)
            self.assertEqual(dp._DPLib__epsilon, case,
                             msg='testing method: `set_global_epsilon_delta`, case: `epsilon_true` failed')
            del dp

        for case in epsilon_false:
            try:
                dp = DPLib()
                dp.set_global_epsilon_delta(epsilon=case)
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertIn(member=(str(exc_type), str(exc_msg)),
                              container=(("<class 'TypeError'>", 'Epsilon and delta must be numeric'),
                                         ("<class 'ValueError'>", 'Epsilon must be non-negative'),
                                         ("<class 'ValueError'>", 'Epsilon and Delta cannot both be zero')),
                              msg='testing method: `set_global_epsilon_delta`, case: `epsilon_false` failed')

                del dp
                continue

        for case in delta_true:
            dp = DPLib(global_epsilon=0.1, global_delta=case)
            self.assertEqual(dp._DPLib__delta, case,
                             msg='testing method: `set_global_epsilon_delta`, case: `delta_true` failed')
            del dp

        for case in delta_false:
            try:
                DPLib(global_epsilon=0, global_delta=case)
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertIn(member=(str(exc_type), str(exc_msg)),
                              container=(("<class 'TypeError'>", 'Epsilon and delta must be numeric'),
                                         ("<class 'ValueError'>", 'Delta must be in range [0, 1]'),
                                         ("<class 'ValueError'>", 'Epsilon and Delta cannot both be zero')),
                              msg='testing method: `set_global_delta, case: `delta_false` failed')

                continue

    def test_set_global_sensitivity(self):
        for case in sensitivity_true:
            dp = DPLib()
            dp.set_global_sensitivity(case)
            self.assertEqual(dp._DPLib__sensitivity, case,
                             msg='testing method: `set_global_sensitivity`, case: `sensitivity_true` failed')
            del dp

        for case in sensitivity_false:
            try:
                dp = DPLib()
                dp.set_global_sensitivity(case)
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertIn(member=(str(exc_type), str(exc_msg)),
                              container=(("<class 'TypeError'>", 'Sensitivity must be numeric'),
                                         ("<class 'ValueError'>", 'Sensitivity must be strictly positive')),
                              msg='testing method: `set_global_sensitivity`, case: `sensitivity_false` failed')
                del dp
                continue

    def test_set_sdf(self):
        dp = DPLib(sdf=sdf_true)
        self.assertEqual(dp.sdf, sdf_true, msg='testing method: `set_sdf`, case: `delta_true` failed')
        del dp

        for case in sdf_false:
            try:
                DPLib(sdf=case)
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertEqual(first=(str(exc_type), str(exc_msg)),
                                 second=("<class 'TypeError'>", 'Given sdf is not a Spark DataFrame'),
                                 msg='testing method: `set_sdf`, case: `sdf_false` failed')
                continue

    def test_set_column(self):
        dp = DPLib()

        with self.assertRaises(ValueError) as cm:
            dp.set_column(column_name='n', category='numeric')
        self.assertEqual(str(cm.exception), 'Add an eligible Spark DataFrame before adding columns',
                         msg='testing method: `set_column`, case: `add_sdf_false` failed')

        dp.set_sdf(sdf_true)
        column_name_false = [12, 2 + 4j, dp, sdf_true, 'string', 'column1']
        for case in column_name_false:
            with self.assertRaises(ValueError) as cm:
                dp.set_column(column_name=case, category='numeric')
            self.assertEqual(str(cm.exception), 'Cannot find column in given DataFrame',
                             msg='testing method: `set_column`, case: `column_name_false` failed')

        category_false = column_name_false + ['n', 'b', 'number', 'bool', 'integer']
        for case in category_false:
            with self.assertRaises(ValueError) as cm:
                dp.set_column(column_name='n', category=case)
            self.assertEqual(str(cm.exception), 'category must be either `numeric` or `boolean`',
                             msg='testing method: `set_column`, case: `category_false` failed')

        with self.assertRaises(ValueError) as cm:
            dp.set_column(column_name='n', category='numeric')
        self.assertEqual(str(cm.exception), 'Epsilon must be set',
                         msg='testing method: `set_column`, case: `epsilon=None_false` failed')

        for case in [val for val in epsilon_false if val is not None]:
            try:
                dp.set_column(column_name='n', category='numeric', epsilon=case)
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertIn(member=(str(exc_type), str(exc_msg)),
                              container=(("<class 'TypeError'>", 'Epsilon and delta must be numeric'),
                                         ("<class 'ValueError'>", 'Epsilon must be non-negative'),
                                         ("<class 'ValueError'>", 'Epsilon and Delta cannot both be zero')),
                              msg='testing method: `set_column`, case: `epsilon_false` failed')
                continue

        for case in delta_false:
            try:
                dp.set_column(column_name='n', category='numeric', epsilon=0, delta=case)
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertIn(member=(str(exc_type), str(exc_msg)),
                              container=(("<class 'TypeError'>", 'Epsilon and delta must be numeric'),
                                         ("<class 'ValueError'>", 'Delta must be in range [0, 1]'),
                                         ("<class 'ValueError'>", 'Epsilon and Delta cannot both be zero')),
                              msg='testing method: `set_column`, case: `delta_false` failed')

                continue

        for case in sensitivity_false:
            try:
                dp.set_column(column_name='n', category='numeric', epsilon=0.00001, sensitivity=case)
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertIn(member=(str(exc_type), str(exc_msg)),
                              container=(("<class 'TypeError'>", 'Sensitivity must be numeric'),
                                         ("<class 'ValueError'>", 'Sensitivity must be strictly positive')),
                              msg='testing method: `set_column`, case: `sensitivity_false` failed')
                continue

        for case in range(len(lower_bound_false)):
            try:
                dp.set_column(column_name='n', category='numeric', epsilon=0.00001, sensitivity=10,
                              lower_bound=lower_bound_false[case], upper_bound=upper_bound_false[case])
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertIn(member=(str(exc_type), str(exc_msg)),
                              container=(("<class 'TypeError'>", 'Bounds must be numeric'),
                                         ("<class 'ValueError'>", 'Lower bound must not be greater than upper bound')),
                              msg='testing method: `set_column`, case: `bound_false` failed')
                continue

        for case in round_false:
            try:
                dp.set_column(column_name='n', category='numeric', epsilon=0.00001, sensitivity=10,
                              round=case)
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertEqual(first=(str(exc_type), str(exc_msg)),
                                 second=("<class 'TypeError'>", 'round must be positive integer'),
                                 msg='testing method: `set_column`, case: `round_false` failed')
                continue

        for case in range(len(label1_false)):
            try:
                dp.set_column(column_name='b', category='boolean', epsilon=0.00001,
                              label1=label1_false[case], label2=label2_false[case])
            except Exception:

                exc_type, exc_msg, traceback = sys.exc_info()
                self.assertIn(member=(str(exc_type), str(exc_msg)),
                              container=(("<class 'TypeError'>", 'Labels must be strings.'),
                                         ("<class 'ValueError'>", 'Labels must be non-empty strings'),
                                         ("<class 'ValueError'>", 'Labels must not match')),
                              msg='testing method: `set_column`, case: `label_false` failed')
                continue

        for case in label_sdf_false:
            dp.set_sdf(case)

            with self.assertRaises(ValueError) as cm:
                dp.set_column(column_name='b', category='boolean', epsilon=0.00001,
                              label1='yes', label2='no')
            self.assertEqual(str(cm.exception), 'Labels in column `b` does not match with labels entered',
                             msg='testing method: `set_column`, case: `label_sdf_false` failed')
            continue

        del dp

        dp = DPLib(global_epsilon=0.00001, sdf=set_column_sdf)
        dp.set_global_sensitivity(sensitivity=10)
        dp.set_column(column_name='n', category='numeric')
        dp.set_column(column_name='rn', category='numeric',
                      lower_bound=round(20000), upper_bound=round(80000),
                      round=2)
        dp.set_column(column_name='b', category='boolean',
                      label1='yes', label2='no')
        self.assertDictEqual(dp._DPLib__columns, set_column_true,
                             msg='testing method: `set_column`, case: `set_column_true` failed')
        del dp

    def test_drop_column(self):
        dp = DPLib(global_epsilon=0.00001, sdf=sdf_true)
        dp.set_global_sensitivity(sensitivity=10)

        dp.set_column(column_name='n', category='numeric')
        dp.set_column(column_name='rn', category='numeric',
                      lower_bound=round(20000), upper_bound=round(80000),
                      round=2)
        dp.set_column(column_name='b', category='boolean',
                      label1='yes', label2='no')

        dp.drop_column('n')
        self.assertDictEqual(dp._DPLib__columns, drop_column_true,
                             msg='testing method: `drop_column`, case: `drop_column_true` failed')

        dp.drop_column('*')
        self.assertDictEqual(dp._DPLib__columns, {},
                             msg='testing method: `drop_column`, case: `drop_column_all` failed')

        del dp

    def test_get_config(self):
        dp = DPLib(global_epsilon=0.00001, sdf=sdf_true)
        dp.set_global_sensitivity(sensitivity=10)

        dp.set_column(column_name='n', category='numeric')
        dp.set_column(column_name='rn', category='numeric',
                      lower_bound=round(20000), upper_bound=round(80000),
                      round=2)
        dp.set_column(column_name='b', category='boolean',
                      label1='yes', label2='no')

        io_Obj = io.StringIO()
        with contextlib.redirect_stdout(io_Obj):
            dp.get_config()
        config_str = io_Obj.getvalue()
        self.assertEqual(config_str, get_config_true,
                         msg='testing method: `get_config`, case: `get_config_true` failed')

    def test_execute(self):
        dp = DPLib(global_epsilon=0.00001, sdf=sdf_true)
        dp.set_global_sensitivity(sensitivity=10)

        with self.assertRaises(ValueError) as cm:
            dp.execute()
        self.assertEqual(str(cm.exception), 'No columns added for execution',
                         msg='testing method: `execute`, case: `execute_false` failed')

        del dp

    def test_execution_speed(self):

        dp = DPLib(global_epsilon=0.00001, sdf=set_column_sdf)
        dp.set_global_sensitivity(sensitivity=10)

        dp.set_column(column_name='n', category='numeric')
        dp.set_column(column_name='rn', category='numeric',
                      lower_bound=round(20000), upper_bound=round(80000),
                      round=2)
        dp.set_column(column_name='b', category='boolean',
                      label1='yes', label2='no')

        start_time = time.time_ns()
        dp.execute()
        finish_time = time.time_ns()

        self.assertLessEqual((finish_time - start_time) / set_column_sdf.count(), 2000,
                             msg='testing execution speed, test: failed')

        del dp


if __name__ == '__main__':
    def generate_rand_tuple(num_range=100000, choice_list=('yes', 'no')):
        number_1 = randint(0, num_range) + random()
        number_2 = randint(0, num_range) + random()
        string = choice(choice_list)
        return number_1, number_2, string


    spark = SparkSession.builder \
        .master('local') \
        .appName('differential_privacy') \
        .config('spark.some.config.option', 'some-value') \
        .getOrCreate()

    spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

    schema = StructType([
        StructField('n', DoubleType()),
        StructField('rn', DoubleType()),
        StructField('b', StringType())
    ])

    epsilon_true = [0, 0.00000001, 1000, 100000.000001, float('inf')]
    epsilon_false = [0, -4, -5.5, float('-inf'), 3 + 6j, -4j, 'string', None]

    delta_true = [0.0000000000000000001, 0.9999999999999999, 0, 1]
    delta_false = [0, -0.00000001, 1.000000000000001, -4, 'string', None]

    sdf_true = spark.createDataFrame(data=[generate_rand_tuple() for _ in range(10)], schema=schema)
    sdf_false = [[1, 2, 3], (1, 2, 3), 'string', 5.7, 4 + 6j, sdf_true.rdd]

    sensitivity_true = [0.0000000000000000001, float('inf')]
    sensitivity_false = [0, -1.5, 3 - 5j, float('-inf'), 'string', None]

    lower_bound_false = [4 - 2j, 200, float('inf'), 45, 'string', spark]
    upper_bound_false = [4 + 2j, 199.9999999999999999, float('inf'), -45, 'string', spark]

    round_false = [0.0000000000000000001, 20.001, -5, -0.9999, float('inf')]

    label1_false = ['yes', 'no', 1, sdf_true, float('inf'), '']
    label2_false = ['yes', 'no', -3, sdf_true.rdd, float('-inf'), '']
    label_sdf_false = [spark.createDataFrame(
        data=[generate_rand_tuple(choice_list=('male', 'female', 'undefined')) for _ in range(100)], schema=schema),
        spark.createDataFrame(
            data=[generate_rand_tuple(choice_list=('yes', 'no', 'can\'t say for sure', 'I don\'t know'))
                  for _ in range(100)], schema=schema),
        spark.createDataFrame(data=[generate_rand_tuple(choice_list=('1', 'no')) for _ in range(100)],
                              schema=schema),
        spark.createDataFrame(data=[generate_rand_tuple(choice_list=('yes', '2')) for _ in range(100)],
                              schema=schema)]
    set_column_sdf = spark.createDataFrame(data=[generate_rand_tuple() for _ in range(100000)], schema=schema)
    set_column_true = json.loads('{"n": {"category": "numeric", "epsilon": 1e-05, "delta": 0.0, "sensitivity": 10.0, '
                                 '"lower_bound": -Infinity, "upper_bound": Infinity}, "rn": {"category": "numeric", '
                                 '"epsilon": 1e-05, "delta": 0.0, "sensitivity": 10.0, "lower_bound": 20000, '
                                 '"upper_bound": 80000, "round": 2}, "b": {"category": "boolean", "epsilon": 1e-05, '
                                 '"delta": 0.0, "label1": "yes", "label2": "no"}}')

    drop_column_true = json.loads('{"rn": {"category": "numeric", "epsilon": 1e-05, "delta": 0.0, "sensitivity": '
                                  '10.0, "lower_bound": 20000, "upper_bound": 80000, "round": 2}, "b": {"category": '
                                  '"boolean", "epsilon": 1e-05, "delta": 0.0, "label1": "yes", "label2": "no"}}')

    get_config_true = 'Global parameters\n-----------------\n\nEpsilon      1e-05\nDelta        0.0\nSensitivity  10.0\n\n\nColumn specific parameters\n--------------------------\n\n| Column name   | Column category   | Epsilon   | Delta   | Sensitivity   | Lower bound   | Upper bound   | Round   | Label 1   | Label 2   |\n|---------------|-------------------|-----------|---------|---------------|---------------|---------------|---------|-----------|-----------|\n| n             | numeric           | 1e-05     | 0.0     | 10.0          | -inf          | inf           | --      | --        | --        |\n| rn            | numeric           | 1e-05     | 0.0     | 10.0          | 20000         | 80000         | 2       | --        | --        |\n| b             | boolean           | 1e-05     | 0.0     | --            | --            | --            | --      | yes       | no        |\n'

    unittest.main()

    spark.stop()
