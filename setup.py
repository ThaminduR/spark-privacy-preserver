import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spark_privacy_preserver",  # Replace with your own username
    version="0.3.1",
    author="thamindu",
    author_email="thamindu.randil@gmail.com",
    description="Anonymizing Library for Apache Spark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThaminduR/spark-privacy-preserver",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas>=1.1',
        'pyspark==2.4.5',
        'pyarrow==0.17.1',
        'diffprivlib==0.2.1',
        'tabulate==0.8.7',
        'mypy>=0.770',
        'kmodes'
    ],
    extras_requires={
        'DPLib': ['notebook']
    }
)