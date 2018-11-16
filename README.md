[![PyPI Shield](https://img.shields.io/pypi/v/piex.svg)](https://pypi.python.org/pypi/piex)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/piex.svg?branch=master)](https://travis-ci.org/HDI-Project/piex)

# Pipeline Explorer

Classes and functions to explore and reproce the performance obtained by
thousands of MLBlocks pipelines and templates accross hundreds of datasets.

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/piex
- Homepage: https://github.com/HDI-Project/piex


# Getting Started

## Installation

```bash
$ git clone git@github.com:HDI-Project/piex.git
$ cd piex
$ pip install -e .
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
available datasets.


```python
datasets = piex.get_datasets()
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

The list of that have been executed can be obtained with the method `get_tests`.

Just like the `get_datasets`, any keyword arguments will be used to filter the results.


```python
import pandas as pd

tests = piex.get_tests()
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

Just like the best pipeline, the best tempalte for a given dataset can be obtained using
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



Or obtaning the corresponding tunable ranges, ready to be used with a tuner:


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
the cross validation specification.

If no hyperparameters are passed, the default ones will be used:


```python
piex.score_template(template_name, 'uu4_SPECT', n_splits=3, random_state=43)
```




    (0.8555346666968675, 0.028343173498423108)



However, different hyperparameters can be passed as a dictionary:


```python
hyperparameters = piex.get_default_hyperparameters(template_name)
hyperparameters['xgboost.XGBClassifier#1']['learning_rate'] = 1

piex.score_template(template_name, 'uu4_SPECT', hyperparameters, n_splits=3, random_state=43)
```




    (0.8754554700753094, 0.019151608028236813)


