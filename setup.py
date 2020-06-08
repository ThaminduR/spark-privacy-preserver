import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spark_privacy_preserver",  # Replace with your own username
    version="0.0.1",
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
)