
<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>


[![PyPI Shield](https://img.shields.io/pypi/v/piex.svg)](https://pypi.python.org/pypi/piex)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/piex.svg?branch=master)](https://travis-ci.org/HDI-Project/piex)

# Pipeline Explorer

Classes and functions to explore and reproduce the performance obtained by
thousands of MLBlocks pipelines and templates across hundreds of datasets.

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/piex
- Homepage: https://github.com/HDI-Project/piex


# Overview

This repository contains a collection of classes and functions which allows a user to easily
explore the results of a series of experiments run by team MIT using MLBlocks pipelines over
a large collection of Datasets.

Along with this library we are releasing a number of fitted pipelines, their performance on
cross validation, test data and metrics. The results of these experiments were stored in a
Database and later on uploaded to Amazon S3, from where they can be downloaded and analyzed
using the Pipeline Explorer.

We will continuously add more pipelines, templates and datasets to our experiments and make
them publicly available to the community.

These can be used for the following purposes:

* Find what is the best score we found so far for a given dataset and task type (given the
  search space we defined and our tuners)
* Use information about pipeline performance to do meta learning

Current summary of our experiments is:

<div>
<table class="dataframe">
  <thead>
    <tr>
      <th># of</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>datasets</th>
      <td>453</td>
    </tr>
    <tr>
      <th>pipelines</th>
      <td>2115907</td>
    </tr>
    <tr>
      <th>templates</th>
      <td>63</td>
    </tr>
    <tr>
      <th>tests</th>
      <td>2152</td>
    </tr>
  </tbody>
</table>
</div>


## Concepts

Before diving into the software usage, we briefly explain some concepts and terminology.

### Primitives

We call the smallest computational blocks used in a Machine Learning process
**primitives**, which:

* Can be either classes or functions.
* Have some initialization arguments, which MLBlocks calls `init_params`.
* Have some tunable hyperparameters, which have types and a list or range of valid values.

### Templates

Primitives can be combined to form what we call **Templates**, which:

* Have a list of primitives.
* Have some initialization arguments, which correspond to the initialization arguments
  of their primitives.
* Have some tunable hyperparameters, which correspond to the tunable hyperparameters
  of their primitives.

### Pipelines

Templates can be used to build **Pipelines** by taking and fixing a set of valid
hyperparameters for a Template. Hence, Pipelines:

* Have a list of primitives, which corresponds to the list of primitives of their template.
* Have some initialization arguments, which correspond to the initialization arguments
  of their template.
* Have some hyperparameter values, which fall within the ranges of valid tunable
  hyperparameters of their template.

A pipeline can be fitted and evaluated using the MLPipeline API in MLBlocks.

### Datasets

A collection of ~450 datasets was used covering 6 different data modalities and 17 task types.

Each dataset was split using a holdout method in two parts, training and testing, which were
used respectively to find and fit the optimal pipeline for each dataset, and to later on
evaluate the goodness-of-fit of each pipeline against a specific metric for each dataset.

This collection of datasets is stored in an Amazon S3 Bucket in the [D3M format](https://github.com/mitll/d3m-schema),
including the training and testing partitioning, and can be downloaded both using piex or a web browser following this link: https://d3m-data-dai.s3.amazonaws.com/index.html

### What is an experiment/test?

Throughout our description we will refer to a search process as an **experiment** or a **test**.
An experiment/test is defined as follows:

* It is given a dataset and a task
* It is given a template
* It then searches using a Bayesian tuning algorithm (using a tuner from our BTB library). Tuning
  algorithm tests multiple pipelines derived from the template and tries to find the best set of
  hyperparameters possible for that template on each dataset.
* During the search process, a collection of information is stored in the database and is
  available through piex. They are:
    * Cross Validation score obtained over the training partition by each pipeline fitted during
      the search process.
    * In parallel, at some points in time the best pipeline already found was validated against
      the testing data, and the   obtained score was also stored in the database.

Each experiment was given one or more of the following configuration values:

* Timeout: Maximum time that the search process is allowed to run.
* Budget: Maximum number of tuning iterations that the search process is allowed to perform.
* Checkpoints: List of points in time, in seconds, where the best pipeline so far was
  scored against the testing data.
* Pipeline: The name of the template to use to build the pipelines.
* Tuner Type: The type of tuner to use, `gp` or `uniform`.


# Getting Started

## Installation

The simplest and recommended way to install the Pipeline Explorer is using pip:

```bash
pip install piex
```

Alternatively, you can also clone the repository and install it from sources

```bash
git clone git@github.com:HDI-Project/piex.git
cd piex
pip install -e .
```

# Usage

## The S3PipelineExplorer

The **S3PipelineExplorer** class provides methods to download the results from previous
tests executions from S3, see which pipelines obtained the best scores and load them
as a dictionary, ready to be used by an MLPipeline.

To start working with it, it needs to be given the name of the S3 Bucket from which
the data will be downloaded.

For this examples, we will be using the `ml-pipelines-2018` bucket, where the results
of the experiments run for the Machine Learning Bazaar paper can be found.


```python
from piex.explorer import S3PipelineExplorer

piex = S3PipelineExplorer('ml-pipelines-2018')
```

### The Datasets

The `get_datasets` method returns a `pandas.DataFrame` with information about the
available datasets, their data modalities, task types and task subtypes.


```python
datasets = piex.get_datasets()
datasets.shape
```




    (453, 4)




```python
datasets.head()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>data_modality</th>
      <th>task_type</th>
      <th>task_subtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>314</th>
      <td>124_120_mnist</td>
      <td>image</td>
      <td>classification</td>
      <td>multi_class</td>
    </tr>
    <tr>
      <th>315</th>
      <td>124_138_cifar100</td>
      <td>image</td>
      <td>classification</td>
      <td>multi_class</td>
    </tr>
    <tr>
      <th>316</th>
      <td>124_153_svhn_cropped</td>
      <td>image</td>
      <td>classification</td>
      <td>multi_class</td>
    </tr>
    <tr>
      <th>317</th>
      <td>124_174_cifar10</td>
      <td>image</td>
      <td>classification</td>
      <td>multi_class</td>
    </tr>
    <tr>
      <th>318</th>
      <td>124_178_coil100</td>
      <td>image</td>
      <td>classification</td>
      <td>multi_class</td>
    </tr>
  </tbody>
</table>
</div>




```python
datasets = piex.get_datasets(data_modality='multi_table', task_type='regression')
datasets.head()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>data_modality</th>
      <th>task_type</th>
      <th>task_subtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>311</th>
      <td>uu2_gp_hyperparameter_estimation</td>
      <td>multi_table</td>
      <td>regression</td>
      <td>multivariate</td>
    </tr>
    <tr>
      <th>312</th>
      <td>uu3_world_development_indicators</td>
      <td>multi_table</td>
      <td>regression</td>
      <td>univariate</td>
    </tr>
  </tbody>
</table>
</div>



### The Experiments

The list of tests that have been executed can be obtained with the method `get_tests`.

This method returns a `pandas.DataFrame` that contains a row for each experiment that has been run on each dataset.
This dataset includes information about the dataset, the configuration used for the experiment, such as the
template, the checkpoints or the budget, and information about the execution, such as the timestamp, the exact
software version, the host that executed the test and whether there was an error or not.

Just like the `get_datasets`, any keyword arguments will be used to filter the results.


```python
import pandas as pd

tests = piex.get_tests()
tests.head().T
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>budget</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>checkpoints</th>
      <td>[900, 1800, 3600, 7200]</td>
      <td>[900, 1800, 3600, 7200]</td>
      <td>[900, 1800, 3600, 7200]</td>
      <td>[900, 1800, 3600, 7200]</td>
      <td>[900, 1800, 3600, 7200]</td>
    </tr>
    <tr>
      <th>commit</th>
      <td>4c7c29f</td>
      <td>4c7c29f</td>
      <td>4c7c29f</td>
      <td>4c7c29f</td>
      <td>4c7c29f</td>
    </tr>
    <tr>
      <th>dataset</th>
      <td>196_autoMpg</td>
      <td>26_radon_seed</td>
      <td>LL0_1027_esl</td>
      <td>LL0_1028_swd</td>
      <td>LL0_1030_era</td>
    </tr>
    <tr>
      <th>docker</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>error</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>hostname</th>
      <td>ec2-52-14-97-167.us-east-2.compute.amazonaws.com</td>
      <td>ec2-18-223-109-53.us-east-2.compute.amazonaws.com</td>
      <td>ec2-18-217-79-23.us-east-2.compute.amazonaws.com</td>
      <td>ec2-18-217-239-54.us-east-2.compute.amazonaws.com</td>
      <td>ec2-18-225-32-252.us-east-2.compute.amazonaws.com</td>
    </tr>
    <tr>
      <th>image</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>insert_ts</th>
      <td>2018-10-24 20:05:01.872</td>
      <td>2018-10-24 20:05:02.778</td>
      <td>2018-10-24 20:05:02.879</td>
      <td>2018-10-24 20:05:02.980</td>
      <td>2018-10-24 20:05:03.081</td>
    </tr>
    <tr>
      <th>pipeline</th>
      <td>categorical_encoder/imputer/standard_scaler/xg...</td>
      <td>categorical_encoder/imputer/standard_scaler/xg...</td>
      <td>categorical_encoder/imputer/standard_scaler/xg...</td>
      <td>categorical_encoder/imputer/standard_scaler/xg...</td>
      <td>categorical_encoder/imputer/standard_scaler/xg...</td>
    </tr>
    <tr>
      <th>status</th>
      <td>done</td>
      <td>done</td>
      <td>done</td>
      <td>done</td>
      <td>done</td>
    </tr>
    <tr>
      <th>test_id</th>
      <td>20181024200501872083</td>
      <td>20181024200501872083</td>
      <td>20181024200501872083</td>
      <td>20181024200501872083</td>
      <td>20181024200501872083</td>
    </tr>
    <tr>
      <th>timeout</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>traceback</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>tuner_type</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>update_ts</th>
      <td>2018-10-24 22:05:55.386</td>
      <td>2018-10-24 22:05:57.508</td>
      <td>2018-10-24 22:05:56.337</td>
      <td>2018-10-24 22:05:56.112</td>
      <td>2018-10-24 22:05:56.164</td>
    </tr>
    <tr>
      <th>data_modality</th>
      <td>single_table</td>
      <td>single_table</td>
      <td>single_table</td>
      <td>single_table</td>
      <td>single_table</td>
    </tr>
    <tr>
      <th>task_type</th>
      <td>regression</td>
      <td>regression</td>
      <td>regression</td>
      <td>regression</td>
      <td>regression</td>
    </tr>
    <tr>
      <th>task_subtype</th>
      <td>univariate</td>
      <td>univariate</td>
      <td>univariate</td>
      <td>univariate</td>
      <td>univariate</td>
    </tr>
    <tr>
      <th>metric</th>
      <td>meanSquaredError</td>
      <td>rootMeanSquaredError</td>
      <td>meanSquaredError</td>
      <td>meanSquaredError</td>
      <td>meanSquaredError</td>
    </tr>
    <tr>
      <th>dataset_id</th>
      <td>196_autoMpg_dataset_TRAIN</td>
      <td>26_radon_seed_dataset_TRAIN</td>
      <td>LL0_1027_esl_dataset_TRAIN</td>
      <td>LL0_1028_swd_dataset_TRAIN</td>
      <td>LL0_1030_era_dataset_TRAIN</td>
    </tr>
    <tr>
      <th>problem_id</th>
      <td>196_autoMpg_problem_TRAIN</td>
      <td>26_radon_seed_problem_TRAIN</td>
      <td>LL0_1027_esl_problem_TRAIN</td>
      <td>LL0_1028_swd_problem_TRAIN</td>
      <td>LL0_1030_era_problem_TRAIN</td>
    </tr>
    <tr>
      <th>target</th>
      <td>class</td>
      <td>log_radon</td>
      <td>out1</td>
      <td>Out1</td>
      <td>out1</td>
    </tr>
    <tr>
      <th>size</th>
      <td>24</td>
      <td>160</td>
      <td>16</td>
      <td>52</td>
      <td>32</td>
    </tr>
    <tr>
      <th>size_human</th>
      <td>24K</td>
      <td>160K</td>
      <td>16K</td>
      <td>52K</td>
      <td>32K</td>
    </tr>
    <tr>
      <th>test_features</th>
      <td>7</td>
      <td>28</td>
      <td>4</td>
      <td>10</td>
      <td>4</td>
    </tr>
    <tr>
      <th>test_samples</th>
      <td>100</td>
      <td>183</td>
      <td>100</td>
      <td>199</td>
      <td>199</td>
    </tr>
    <tr>
      <th>train_features</th>
      <td>7</td>
      <td>28</td>
      <td>4</td>
      <td>10</td>
      <td>4</td>
    </tr>
    <tr>
      <th>train_samples</th>
      <td>298</td>
      <td>736</td>
      <td>388</td>
      <td>801</td>
      <td>801</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(tests.groupby(['data_modality', 'task_type']).size(), columns=['count'])
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>data_modality</th>
      <th>task_type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">graph</th>
      <th>community_detection</th>
      <td>5</td>
    </tr>
    <tr>
      <th>graph_matching</th>
      <td>18</td>
    </tr>
    <tr>
      <th>link_prediction</th>
      <td>2</td>
    </tr>
    <tr>
      <th>vertex_nomination</th>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">image</th>
      <th>classification</th>
      <td>57</td>
    </tr>
    <tr>
      <th>regression</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">multi_table</th>
      <th>classification</th>
      <td>1</td>
    </tr>
    <tr>
      <th>regression</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">single_table</th>
      <th>classification</th>
      <td>1405</td>
    </tr>
    <tr>
      <th>collaborative_filtering</th>
      <td>1</td>
    </tr>
    <tr>
      <th>regression</th>
      <td>430</td>
    </tr>
    <tr>
      <th>time_series_forecasting</th>
      <td>175</td>
    </tr>
    <tr>
      <th>text</th>
      <th>classification</th>
      <td>17</td>
    </tr>
    <tr>
      <th>timeseries</th>
      <th>classification</th>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>




```python
tests = piex.get_tests(data_modality='graph', task_type='link_prediction')
tests[['dataset', 'pipeline', 'checkpoints', 'test_id']]
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>pipeline</th>
      <th>checkpoints</th>
      <th>test_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1716</th>
      <td>59_umls</td>
      <td>NaN</td>
      <td>[900, 1800, 3600, 7200]</td>
      <td>20181031040541366347</td>
    </tr>
    <tr>
      <th>2141</th>
      <td>59_umls</td>
      <td>graph/link_prediction/random_forest_classifier</td>
      <td>[900, 1800, 3600, 7200]</td>
      <td>20181031182305995728</td>
    </tr>
  </tbody>
</table>
</div>



### The Experiment Results

The results of the experiments can be seen using the `get_experiment_results` method.

These results include both the cross validation score obtained by the pipeline during
the tuning, as well as the score obtained by this pipeline once it has been fitted
using the training data and then used to make predictions over the test data.

Just like the `get_datasets`, any keyword arguments will be used to filter the results,
including the `test_id`.


```python
results = piex.get_test_results(test_id='20181031182305995728')
results[['test_id', 'pipeline', 'score', 'cv_score', 'elapsed', 'iterations']]
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test_id</th>
      <th>pipeline</th>
      <th>score</th>
      <th>cv_score</th>
      <th>elapsed</th>
      <th>iterations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7464</th>
      <td>20181031182305995728</td>
      <td>graph/link_prediction/random_forest_classifier</td>
      <td>0.499853</td>
      <td>0.843175</td>
      <td>900.255511</td>
      <td>435.0</td>
    </tr>
    <tr>
      <th>7465</th>
      <td>20181031182305995728</td>
      <td>graph/link_prediction/random_forest_classifier</td>
      <td>0.499853</td>
      <td>0.854603</td>
      <td>1800.885417</td>
      <td>805.0</td>
    </tr>
    <tr>
      <th>7466</th>
      <td>20181031182305995728</td>
      <td>graph/link_prediction/random_forest_classifier</td>
      <td>0.499853</td>
      <td>0.854603</td>
      <td>3600.005072</td>
      <td>1432.0</td>
    </tr>
    <tr>
      <th>7467</th>
      <td>20181031182305995728</td>
      <td>graph/link_prediction/random_forest_classifier</td>
      <td>0.785568</td>
      <td>0.860000</td>
      <td>7200.225256</td>
      <td>2366.0</td>
    </tr>
  </tbody>
</table>
</div>



### The Best Pipeline

Information about the best pipeline for a dataset can be obtained using the `get_best_pipeline` method.

This method returns a `pandas.Series` object with information about the pipeline that obtained the
best cross validation score during the tuning, as well as the template that was used to build it.

**Note**: This call will download some data in the background the first time that it is run,
so it might take a while to return.


```python
piex.get_best_pipeline('185_baseball')
```




    id                            17385666-31da-4b6e-ab7f-8ac7080a4d55
    dataset                                 185_baseball_dataset_TRAIN
    metric                                                     f1Macro
    name             categorical_encoder/imputer/standard_scaler/xg...
    rank                                                      0.307887
    score                                                     0.692113
    template                                  5bd0ce5249e71569e8bf8003
    test_id                                       20181024234726559170
    pipeline         categorical_encoder/imputer/standard_scaler/xg...
    data_modality                                         single_table
    task_type                                           classification
    Name: 1149699, dtype: object



Apart from obtaining this information, we can use the `load_best_pipeline` method
to load its JSON specification, ready to be using in an `mlblocks.MLPipeline` object.


```python
pipeline = piex.load_best_pipeline('185_baseball')
pipeline['primitives']
```




    ['mlprimitives.feature_extraction.CategoricalEncoder',
     'sklearn.preprocessing.Imputer',
     'sklearn.preprocessing.StandardScaler',
     'mlprimitives.preprocessing.ClassEncoder',
     'xgboost.XGBClassifier',
     'mlprimitives.preprocessing.ClassDecoder']



### The Best Template

Just like the best pipeline, the best template for a given dataset can be obtained using
the `get_best_template` method.

This returns just the name of the template that was used to build the best pipeline.


```python
template_name = piex.get_best_template('185_baseball')
template_name
```




    'categorical_encoder/imputer/standard_scaler/xgbclassifier'



This can be later on used to explore the template, obtaining its default hyperparameters:


```python
defaults = piex.get_default_hyperparameters(template_name)
defaults
```




    {'mlprimitives.feature_extraction.CategoricalEncoder#1': {'copy': True,
      'features': 'auto',
      'max_labels': 0},
     'sklearn.preprocessing.Imputer#1': {'missing_values': 'NaN',
      'axis': 0,
      'copy': True,
      'strategy': 'mean'},
     'sklearn.preprocessing.StandardScaler#1': {'with_mean': True,
      'with_std': True},
     'mlprimitives.preprocessing.ClassEncoder#1': {},
     'xgboost.XGBClassifier#1': {'n_jobs': -1,
      'n_estimators': 100,
      'max_depth': 3,
      'learning_rate': 0.1,
      'gamma': 0,
      'min_child_weight': 1},
     'mlprimitives.preprocessing.ClassDecoder#1': {}}



Or obtaining the corresponding tunable ranges, ready to be used with a tuner:


```python
tunable = piex.get_tunable_hyperparameters(template_name)
tunable
```




    {'mlprimitives.feature_extraction.CategoricalEncoder#1': {'max_labels': {'type': 'int',
       'default': 0,
       'range': [0, 100]}},
     'sklearn.preprocessing.Imputer#1': {'strategy': {'type': 'str',
       'default': 'mean',
       'values': ['mean', 'median', 'most_frequent']}},
     'sklearn.preprocessing.StandardScaler#1': {'with_mean': {'type': 'bool',
       'default': True},
      'with_std': {'type': 'bool', 'default': True}},
     'mlprimitives.preprocessing.ClassEncoder#1': {},
     'xgboost.XGBClassifier#1': {'n_estimators': {'type': 'int',
       'default': 100,
       'range': [10, 1000]},
      'max_depth': {'type': 'int', 'default': 3, 'range': [3, 10]},
      'learning_rate': {'type': 'float', 'default': 0.1, 'range': [0, 1]},
      'gamma': {'type': 'float', 'default': 0, 'range': [0, 1]},
      'min_child_weight': {'type': 'int', 'default': 1, 'range': [1, 10]}},
     'mlprimitives.preprocessing.ClassDecoder#1': {}}



# Scoring Templates and Pipelines

The **S3PipelineExplorer** class also allows cross validating templates and pipelines
over any of the datasets.

## Scoring a Pipeline

The simplest use case is cross validating a pipeline over a dataset.
For this, we must pass the ID of the pipeline and the name of the dataset to the method `score_pipeline`.

The dataset can be the one that was used during the experiments or a different one.


```python
piex.score_pipeline(pipeline['id'], '185_baseball')
```




    (0.6921128080904511, 0.09950216269594728)




```python
piex.score_pipeline(pipeline['id'], 'uu4_SPECT')
```




    (0.8897656842904123, 0.037662864373452655)



Optionally, the cross validation configuration can be changed


```python
piex.score_pipeline(pipeline['id'], 'uu4_SPECT', n_splits=3, random_state=43)
```




    (0.8869488536155202, 0.019475563687443638)



## Scoring a Template

A Template can also be tested over any dataset by passing its name, the dataset and, optionally,
the cross validation specification. You have to make sure to choose template that is relevant for
the task/data modality for which you want to use it.

If no hyperparameters are passed, the default ones will be used:


```python
piex.score_template(template_name, 'uu4_SPECT', n_splits=3, random_state=43)
```




    (0.8555346666968675, 0.028343173498423108)



You can get the default hyperparameters, and update the hyperparameters by setting values
in the dictionary:

**With this anyone can tune the templates that we have for different task/data modality
types using their own AutoML routine. If you choose to do so, let us know the score you
are getting and the pipeline and we will add to our database.**


```python
hyperparameters = piex.get_default_hyperparameters(template_name)
hyperparameters['xgboost.XGBClassifier#1']['learning_rate'] = 1

piex.score_template(template_name, 'uu4_SPECT', hyperparameters, n_splits=3, random_state=43)
```




    (0.8754554700753094, 0.019151608028236813)
