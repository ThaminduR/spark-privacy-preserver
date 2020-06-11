from pyspark.sql.functions import udf  # type: ignore
from pyspark.sql.types import DoubleType, StringType  # type: ignore
from diffprivlib.mechanisms import LaplaceTruncated, Binary  # type: ignore
from tabulate import tabulate  # type: ignore

# Following imports provide type checking functionality to the library
from pyspark.sql.dataframe import DataFrame as SparkDataFrame  # type: ignore
from numbers import Real  # type: ignore
from typing import List, Dict, Union, Optional  # type: ignore
import sys  # type: ignore

if sys.version_info >= (3, 8):
    from typing import TypedDict  # type: ignore
else:
    from typing_extensions import TypedDict  # type: ignore


class Column(TypedDict, total=False):
    column_name: str
    category: str
    epsilon: Union[int, float]
    delta: Union[int, float]
    sensitivity: Union[int, float, None]
    lower_bound: Union[int, float]
    upper_bound: Union[int, float]
    round: int
    label1: Optional[str]
    label2: Optional[str]


class DPLib:
    r""" Create a differentially private Spark DataFrame from an existing Spark DataFrame

    This class makes use of the 'diffprivlib' library by IBM to create differentially private Spark DataFrame.
    It can handle columns with two major data types: 'numeric' and 'boolean'
        'numeric': column should only have numbers or 'null' value
                    utilizes LaplaceTruncated mechanism from diffprivlib.mechanisms

        'boolean': each row of column should have one of two boolean values set by user
                    utilizes Binary mechanism from diffprivlib.mechanisms

    Attributes:
        sdf: A Spark DataFrame object. Methods: `__check_labels`, `set_column` and `execute`
            will raise ValueError if sdf is None.

    Public methods:
        set_global_epsilon_delta:
            Assigns common epsilon and delta to be used by all columns if they lack
            column specific values for epsilon and delta

        set_global_sensitivity:
            Assigns common sensitivity to be used by all columns if they lack
            column specific value for sensitivity

        set_sdf: Assigns DataFrame object to the class

        set_column: Adds column dictionary to `__columns`

        execute: changes existing sdf to be differentially private. This change is not reversible.

    Examples:
        Check out `differential_preserver_demo.ipynb` Jupyter notebook

    """

    def __init__(self,
                 global_epsilon: Union[int, float, None] = None,
                 global_delta: Union[int, float] = 0.0,
                 sdf: Optional[SparkDataFrame] = None) -> None:
        r""" Inits DPLib

        Args:
            global_epsilon: Common epsilon value to be used by all columns as a fail-safe
            global_delta: Common delta value to be used by all columns as a fail-safe. Defaults to 0.0
            sdf: Spark DataFrame to be converted. Can change to a different DataFrame with set_df()
        """

        self.sdf: Optional[SparkDataFrame] = None
        self.set_sdf(sdf)

        self.__columns: Dict[str, Column] = {}

        self.__epsilon: Optional[float] = None
        self.__delta: Optional[float] = None
        self.__sensitivity: Union[int, float, None] = None

        if global_epsilon is not None and self.__check_epsilon_delta(global_epsilon, global_delta):
            self.__epsilon = float(global_epsilon)
            self.__delta = float(global_delta)

    @staticmethod
    def __check_epsilon_delta(epsilon: Union[int, float], delta: Union[int, float]) -> bool:
        r""" checks whether epsilon and delta meet required conditions for method `execute`

        Always called disregarding category. Both 'LaplaceTruncated' and 'Binary' mechanism
        require epsilon and delta.

        Args:
            epsilon: epsilon value to be used by method `execute`
            delta: delta value to be used by method `execute`

        Returns: True if parameters satisfy the conditions

        Raises: TypeError, ValueError if parameters do not obey the rules

        """
        if not isinstance(epsilon, Real) or not isinstance(delta, Real):
            raise TypeError("Epsilon and delta must be numeric")

        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        if not 0 <= float(delta) <= 1:
            raise ValueError("Delta must be in range [0, 1]")

        if epsilon == 0 and delta == 0:
            raise ValueError("Epsilon and Delta cannot both be zero")

        return True

    @staticmethod
    def __check_sensitivity(sensitivity: Union[int, float, None]) -> bool:
        r""" checks whether sensitivity meets required conditions for method `execute`

        Called only when category = 'numeric'

        Args:
            sensitivity: sensitivity value to be used by method `execute`.

        Returns: True if parameter satisfies the conditions

        Raises: TypeError, ValueError if parameter does not obey the rules

        """
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        return True

    @staticmethod
    def __check_labels(sdf: SparkDataFrame, column_name: str, label1: Optional[str], label2: Optional[str]) -> bool:
        r""" checks whether labels meet required conditions for method `execute`.

        Called only when category = 'boolean'

        Args:
            sdf: Spark DataFrame object
            column_name: specific column which is to be executed using 'Binary' mechanism
            label1: label to be used by 'Binary' mechanism
            label2: label to be used by 'Binary' mechanism

        Returns: True if parameters satisfy the conditions

        Raises: TypeError, ValueError if parameters do not obey the rules

        """
        if not isinstance(label1, str) or not isinstance(label2, str):
            raise TypeError("Labels must be strings.")

        if len(label1) == 0 or len(label2) == 0:
            raise ValueError("Labels must be non-empty strings")

        if label1 == label2:
            raise ValueError("Labels must not match")

        # finds unique values in a column
        labels: List[str] = [row[column_name] for row in sdf.select(column_name).distinct().collect()]

        if len(labels) is not 2 or label1 not in labels or label2 not in labels:
            # checks whether all the rows of column have either label1 or label2
            raise ValueError("Column has multiple unique labels")

        return True

    @staticmethod
    def __check_bounds(lower: Union[int, float], upper: Union[int, float]) -> bool:
        r""" checks whether lower and upper bounds for a specific column bound to their conditions

        Called only when category = 'numeric'

        Args:
            lower: lower bound of a column
            upper: upper bound of a column

        Returns: True if parameters satisfy the conditions

        Raises: TypeError, ValueError if parameters do not obey the rules

        """

        if not isinstance(lower, Real) or not isinstance(upper, Real):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")

        return True

    def set_global_epsilon_delta(self, epsilon: Union[int, float], delta: Union[int, float] = 0.0) -> None:
        r"""  set global epsilon and delta if they satisfy method `__check_epsilon_delta`

        Args:
            epsilon: epsilon value to be used by method `execute`
            delta: delta value to be used by method `execute`

        Raises:
            The method itself will not raise any exceptions.
            However inner method `__check_epsilon_delta` may raise exceptions
        """

        if self.__check_epsilon_delta(epsilon, delta):
            self.__epsilon = float(epsilon)
            self.__delta = float(delta)

    def set_global_sensitivity(self, sensitivity: Union[int, float]) -> None:
        r"""  set global sensitivity if it satisfies method `__check_sensitivity`

        Args:
            sensitivity: sensitivity value to be used by method `execute`.

        Raises:
            The method itself will not raise any exceptions.
            However inner method `__check_sensitivity` may raise exceptions
        """

        if self.__check_sensitivity(sensitivity):
            self.__sensitivity = float(sensitivity)

    def set_sdf(self, sdf: SparkDataFrame) -> None:
        r""" sets Spark DataFrame object to class.

        Args:
            sdf: A Spark DataFrame object

        """

        if isinstance(sdf, SparkDataFrame):
            self.sdf = sdf

    def set_column(self, column_name: str, category: str,
                   epsilon: Union[int, float, None] = None, delta: Union[int, float, None] = None,
                   sensitivity: Union[int, float, None] = None,
                   lower_bound: Union[int, float, None] = None, upper_bound: Union[int, float, None] = None,
                   round: Optional[int] = None,
                   label1: Optional[str] = None, label2: Optional[str] = None) -> None:

        r""" adds a column with custom parameters to the __columns dictionary.

        A column may have specific details. Hence this method allows user to set them individually for a column.
        However in case any one parameter is missing, appropriate value will be copied from available global
        parameters and passed instead.

        Parameters: column_name, category are compulsory. However exceptions may arise when certain optional
        parameters such as lower_bound, upper_bound etc. are not set.

        Args:
            ----------------------- common arguments -----------------------------------------------
            column_name: Name of column
            category: Category the column belongs to: ['numeric', 'boolean']
            epsilon: Epsilon value to be used by method `execute`
            delta: Delta value to be used by method `execute`

            ----------------------- arguments specific to category = 'numeric' ----------------------
            sensitivity: Sensitivity value to be used by method `execute`.
            lower_bound: Lower bound of a column.
            upper_bound: Upper bound of a column.
            round: Rounding factor. Values can be rounded off after applying a certain mechanism.

            ----------------------- arguments specific to category = 'boolean' ----------------------
            label1: label to be used by 'Binary' mechanism.
            label2: label to be used by 'Binary' mechanism.

        Raises: TypeError, ValueError if parameters have not been set correctly.
                Inner methods may raise Exception.

        """

        if self.sdf is None:
            raise ValueError("Add an eligible Spark DataFrame before adding columns")

        if column_name not in self.sdf.columns:
            raise ValueError("Cannot find column in given DataFrame")

        if category not in ['numeric', 'boolean']:
            raise ValueError("Cannot find category in available list")

        column: Column = {'category': category}

        if epsilon is None:
            if self.__epsilon is not None:
                epsilon = self.__epsilon
            else:
                raise ValueError("Epsilon must be set")

        if delta is None:
            if self.__delta is not None:
                delta = self.__delta
            else:
                delta = 0.0

        if epsilon is not None and delta is not None and self.__check_epsilon_delta(epsilon, delta):
            column['epsilon'] = epsilon
            column['delta'] = delta

        if category is 'numeric':

            if sensitivity is None:
                sensitivity = self.__sensitivity
            if self.__check_sensitivity(sensitivity):
                column['sensitivity'] = sensitivity

            if lower_bound is None:
                lower_bound = float('-inf')
            if upper_bound is None:
                upper_bound = float('inf')
            if self.__check_bounds(lower_bound, upper_bound):
                column['lower_bound'] = lower_bound
                column['upper_bound'] = upper_bound

            if round is not None:
                if not isinstance(round, int) or round < 0:
                    raise TypeError("round must be positive integer")
                else:
                    column['round'] = round

        if category is 'boolean':
            if self.__check_labels(self.sdf, column_name, label1, label2):
                column['label1'] = label1
                column['label2'] = label2

        self.__columns[str(column_name)] = column

    def drop_column(self, *columns: str) -> None:
        """ drops a column added using `add_column()` method from `self.__columns`

        Args:
            *columns: one or more column names in string format separated by comma
                        if '*' is given, all columns will be dropped

        """

        if len(columns) == 1 and '*' in columns:
            self.__columns = {}
        else:
            for column in columns:
                if column in self.__columns:
                    del self.__columns[column]

    def get_config(self) -> None:

        print('Global parameters')
        print('-----------------\n')

        global_table: List[List[Union[float, str]]] = [
            ['Epsilon', self.__epsilon if self.__epsilon is not None else '--'],
            ['Delta', self.__delta if self.__delta is not None else '--'],
            ['Sensitivity', self.__sensitivity if self.__sensitivity is not None else '--']
        ]

        print(tabulate(tabular_data=global_table,
                       tablefmt='plain',
                       disable_numparse=True))

        print('\n')
        print('Column specific parameters')
        print('--------------------------\n')

        column_table: List[List] = []

        for column_name, details in self.__columns.items():
            row: List = [column_name,
                         details['category'],
                         details['epsilon'] if 'epsilon' in details else '--',
                         details['delta'] if 'delta' in details else '--',
                         details['sensitivity'] if 'sensitivity' in details else '--',
                         details['lower_bound'] if 'lower_bound' in details else '--',
                         details['upper_bound'] if 'upper_bound' in details else '--',
                         details['round'] if 'round' in details else '--',
                         details['label1'] if 'label1' in details else '--',
                         details['label2'] if 'label2' in details else '--'
                         ]
            column_table.append(row)

        if self.sdf is not None and isinstance(self.sdf, SparkDataFrame):
            for column in self.sdf.columns:
                if column not in self.__columns:
                    row = [column]
                    row += '--' * 9
                    column_table.append(row)

        print(tabulate(tabular_data=column_table,
                       tablefmt='github',
                       disable_numparse=True,
                       headers=['Column name', 'Column category', 'Epsilon', 'Delta', 'Sensitivity', 'Lower bound',
                                'Upper bound', 'Round', 'Label 1', 'Label 2']
                       )
              )

    def execute(self):
        r"""

        Raises:
            The method itself will not raise any exceptions.
            However inner methods may raise Exception, only if parameters have not been set correctly

        """

        laplace: LaplaceTruncated = LaplaceTruncated()
        binary: Binary = Binary()

        for column_name, details in self.__columns.items():

            if details['category'] is 'numeric':

                self.sdf = self.sdf.withColumn(colName=column_name,
                                               col=self.sdf[column_name].cast(DoubleType()))

                laplace.set_epsilon_delta(epsilon=details['epsilon'], delta=details['delta'])
                laplace.set_sensitivity(details['sensitivity'])
                laplace.set_bounds(lower=details['lower_bound'], upper=details['upper_bound'])

                if 'round' in details:

                    def round_randomise(cell):
                        return float(round(laplace.randomise(cell), details['round'])) if cell is not None else None

                    round_randomise_udf = udf(f=round_randomise, returnType=DoubleType())

                    self.sdf = self.sdf.withColumn(colName=column_name,
                                                   col=round_randomise_udf(column_name))

                else:

                    def randomise(cell):
                        return float(laplace.randomise(cell)) if cell is not None else None

                    randomise_udf = udf(f=randomise, returnType=DoubleType())

                    self.sdf = self.sdf.withColumn(colName=column_name,
                                                   col=randomise_udf(column_name))

            elif details['category'] is 'boolean':

                self.sdf = self.sdf.withColumn(colName=column_name,
                                               col=self.sdf[column_name].cast(StringType()))

                binary.set_epsilon_delta(epsilon=details['epsilon'], delta=details['delta'])
                binary.set_labels(value0=details['label1'], value1=details['label2'])

                def randomise(cell):
                    return binary.randomise(cell) if cell is not None else None

                randomise_udf = udf(f=randomise, returnType=StringType())

                self.sdf = self.sdf.withColumn(colName=column_name,
                                               col=randomise_udf(column_name))
