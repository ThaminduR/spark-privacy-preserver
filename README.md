# spark-privacy-preserver

## Introduction to Differential Privacy

Differential privacy is one of the data preservation paradigms similar to K-Anonymity, T-Closeness and L-Diversity.
It alters each value of actual data in a dataset according to specific constraints set by the owner and produces a 
differentially-private dataset. This anonymized dataset is then released for public utilization.

**ε-differential privacy** is one of the methods in differential privacy. Laplace based ε-differential privacy is 
applied in this library. The method states that the randomization should be according the **epsilon** (ε) 
(should be >0) value set by data owner. After randomization a typical noise is added to the dataset. It is calibrated 
according to the **sensitivity** value (λ) set by the data owner. 

Other than above parameters, a third parameter **delta** (δ) is added into the mix to increase accuracy of the 
algorithm. A **scale** is computed from the above three parameters and a new value is computed.

> scale = λ / (ε - _log_(1 - δ))

> random_number = _random_generator_(0, 1) - 0.5

> sign = _get_sign_(random_number)

> new_value = value - scale × sign × _log_(1 - 2 × _mod_(random_number))

In essence above steps mean that a laplace transform is applied to the value and a new value is randomly selected 
according to the parameters. When the scale becomes larger, the deviation from original value will increase.

## Achieving Differential Privacy
    
### Requirements

* Python

Python versions above Python 3.6 and below Python 3.8 are recommended. The module is developed and tested on: 
Python 3.7.7 and pip 20.0.2. (It is better to avoid Python 3.8 as it has some compatibility issues with Spark)

* Apache Spark

Spark 2.4.5 is recommended. 

* Java

Java 8 is recommended. Newer versions of java are incompatible with Spark.

The module is developed and tested on:
    
    java version "1.8.0_231"
    Java(TM) SE Runtime Environment (build 1.8.0_231-b11)
    Java HotSpot(TM) 64-Bit Server VM (build 25.231-b11, mixed mode)

Make sure the following Python packages are installed:
1. PySpark: ```pip install pyspark==2.4.5```
2. PyArrow: ```pip install pyarrow==0.17.1```
3. IBM Differential Privacy Library: ```pip install diffprivlib==0.2.1```
4. MyPy:  ```pip install mypy==0.770```

### Procedure/Demo

Step by step procedure on how to use the module with explanation is given in the following notebook: **example.ipynb**