# -*- coding: utf-8 -*-

import gzip
import io
import json
import logging
import os
import pickle

import boto3
import pandas as pd
from mlblocks import MLPipeline

from piex.scoring import pipeline_dataset_score

LOGGER = logging.getLogger(__name__)


class PipelineExplorer:

    DATASETS_COLUMNS = ['dataset', 'data_modality', 'task_type', 'task_subtype']

    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.csvs_path = os.path.join(data_path, 'csvs')

        if not os.path.exists(self.csvs_path):
            os.makedirs(self.csvs_path)

        self.dfs = dict()

    def get_datasets(self, **filters):
        """Get a pandas DataFrame with information about the available datasets."""
        raise NotImplementedError

    def get_tests(self, **filters):
        """Get a pandas DataFrame with information about the executed tests."""
        raise NotImplementedError

    def get_test_results(self, **filters):
        """Get a pandas DataFrame with the results of the executed tests."""
        raise NotImplementedError

    def get_pipelines(self, **filters):
        """Get a pandas DataFrame with information about the scored pipelines."""
        raise NotImplementedError

    def get_best_pipeline(self, dataset):
        """Get information about the best pipeline ever found for the given dataset."""
        raise NotImplementedError

    def get_templates(self, **filters):
        """Get a pandas DataFrame with information about the available templates."""
        raise NotImplementedError

    def get_dataset_id(self, dataset):
        """Get the TRAIN dataset id for a given dataset."""
        raise NotImplementedError

    def load_pipeline(self, pipeline_id):
        """Get the dict representation of the pipeline."""
        raise NotImplementedError

    def load_template(self, template_name):
        """Get the dict representation of the template."""
        raise NotImplementedError

    def load_best_pipeline(self, dataset):
        """Get the dict representation of the best pipeline ever found for this dataset."""
        pipeline = self.get_best_pipeline(dataset)
        return self.load_pipeline(pipeline.id)

    def get_best_template(self, dataset):
        """Get the name of the template that got the best score for this dataset."""
        pipeline = self.get_best_pipeline(dataset)
        return pipeline['name']

    def load_best_template(self, dataset):
        """Get the dict representation of the best template ever found for this dataset."""
        template_name = self.get_best_template(dataset)
        return self.load_template(template_name)

    def get_default_hyperparameters(self, template_name):
        """Get the default hyperparmeters of the given template."""
        template = self.load_template(template_name)
        mlpipeline = MLPipeline.from_dict(template)
        return mlpipeline.get_hyperparameters()

    def get_tunable_hyperparameters(self, template_name):
        """Get the tunable hyperparmeters of the given template."""
        template = self.load_template(template_name)
        mlpipeline = MLPipeline.from_dict(template)
        return mlpipeline.get_tunable_hyperparameters()

    def score_pipeline(self, pipeline_id, dataset, n_splits=5, cv=None, random_state=0):
        """Cross validate the given pipeline on this dataset."""
        pipeline = self.load_pipeline(pipeline_id)
        return pipeline_dataset_score(pipeline, dataset, n_splits, cv, random_state)

    def score_template(self, template_name, dataset, hyperparameters=None,
                       n_splits=5, cv=None, random_state=0):
        """Cross validate the given template on this dataset."""

        template = self.load_template(template_name)
        if hyperparameters is None:
            hyperparameters = self.get_default_hyperparameters(template_name)

        template['hyperparameters'] = hyperparameters
        return pipeline_dataset_score(template, dataset, n_splits, cv, random_state)


class MongoPipelineExplorer(PipelineExplorer):

    def __init__(self, db, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = db

    def _get_collection(self, collection, query=None, project=None):
        cursor = self.db[collection].find(query or dict(), project)
        return pd.DataFrame(list(cursor))

    def get_datasets(self, **filters):
        df = self._get_collection('datasets', filters).reindex(columns=self.DATASETS_COLUMNS)
        return df.sort_values(self.DATASETS_COLUMNS).dropna(subset=['task_type'])

    def get_tests(self, **filters):
        ddf = self._get_collection('datasets', filters, {'_id': 0})
        datasets = list(ddf.dataset.unique())
        tdf = self._get_collection('tests', {'dataset': {'$in': datasets}}, {'_id': 0})
        return tdf.loc[(tdf[list(filters)] == pd.Series(filters)).all(axis=1)].copy()

    def get_test_results(self, **filters):
        rdf = self._get_collection('test_results', filters, {'_id': 0})
        test_ids = list(rdf.test_id.unique())
        tdf = self._get_collection('tests', {'test_id': {'$in': test_ids}}, {'_id': 0})
        return rdf.merge(tdf, how='left', on=['dataset', 'test_id'], suffixes=('', '_results'))

    def get_pipelines(self, **filters):
        if 'data_modality' in filters:
            filters['loader.data_modality'] = filters.pop('data_modality')
        if 'task_type' in filters:
            filters['loader.task_type'] = filters.pop('task_type')

        project = [
            '_id', 'loader', 'dataset', 'metric', 'name', 'rank',
            'score', 'template', 'test_id', 'pipeline'
        ]

        df = self._get_collection('solutions', filters, project)
        df['pipeline'] = df['name']
        loader = df.pop('loader')
        df['data_modality'] = loader.apply(lambda l: l.get('data_modality'))
        df['task_type'] = loader.apply(lambda l: l.get('task_type'))
        return df

    def get_dataset_id(self, dataset):
        document = self.db.datasets.find_one({'dataset': dataset}, ['dataset_id'])
        if not document:
            raise ValueError('Unknown dataset: {}'.format(dataset))

        return document['dataset_id']

    def get_best_pipeline(self, dataset):
        query = {
            'dataset': {
                '$in': [
                    dataset,
                    self.get_dataset_id(dataset)
                ]
            }
        }
        solutions = self.db.solutions.find(query)
        best = list(solutions.sort('rank', 1).limit(1))
        if best:
            best = self.get_pipelines(_id=best[0]['_id']).iloc[0]
            return best.rename({'_id': 'id'})

    def get_templates(self, **filters):
        templates = list()
        project = [
            '_id',
            'metadata',
            'primitives',
        ]

        for template in self.db.pipelines.find({}, project):
            template.update(template.pop('metadata'))
            template['id'] = str(template.pop('_id'))
            templates.append(template)

        tdf = pd.DataFrame(templates)
        return tdf[[
            'id',
            'name',
            'data_type',
            'task_type',
            'primitives',
            'insert_ts'
        ]]

    def load_pipeline(self, pipeline_id):
        return self.db.solutions.find_one({'_id': pipeline_id})

    def load_template(self, template_name):
        return self.db.templates.find_one({'metadata.name': template_name})


class S3PipelineExplorer(PipelineExplorer):

    def __init__(self, bucket, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket = bucket

    def _get_table(self, table_name):
        LOGGER.info("Downloading %s csv from S3", table_name)
        key = os.path.join('csvs', table_name + '.csv.gz')

        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.bucket, Key=key)

        body_bytes = io.BytesIO(obj['Body'].read())
        gzip_file = gzip.GzipFile(fileobj=body_bytes, mode='rb')

        return pd.read_csv(gzip_file)

    def _load_table(self, table_name):
        df = self.dfs.get(table_name)
        if df is not None:
            return df

        pkl_filename = os.path.join(self.csvs_path, table_name + '.pkl')
        if os.path.exists(pkl_filename):
            with open(pkl_filename, 'rb') as pkl_file:
                df = pickle.load(pkl_file)

        else:
            df = self._get_table(table_name)
            with open(pkl_filename, 'wb') as pkl_file:
                pickle.dump(df, pkl_file)

        self.dfs[table_name] = df

        return df

    def _get_json(self, folder, pipeline_id):
        key = os.path.join(folder, pipeline_id + '.json.gz')
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.bucket, Key=key)

        body_bytes = io.BytesIO(obj['Body'].read())
        gzip_file = gzip.GzipFile(fileobj=body_bytes, mode='rb')
        return json.load(gzip_file)

    def get_tests(self, **filters):
        ddf = self._load_table('datasets')
        tdf = self._load_table('tests')
        tdf = tdf.merge(ddf, how='left', on='dataset')
        return tdf.loc[(tdf[list(filters)] == pd.Series(filters)).all(axis=1)].copy()

    def get_test_results(self, **filters):
        rdf = self._load_table('test_results')
        return rdf.loc[(rdf[list(filters)] == pd.Series(filters)).all(axis=1)]

    def get_pipelines(self, **filters):
        df = self._load_table('pipelines')
        df['pipeline'] = df['name']
        return df.loc[(df[list(filters)] == pd.Series(filters)).all(axis=1)].copy()

    def get_templates(self, **filters):
        df = self._load_table('pipelines')
        df['pipeline'] = df['name']
        return df.loc[(df[list(filters)] == pd.Series(filters)).all(axis=1)].copy()

    def get_datasets(self, **filters):
        df = self._load_table('datasets').reindex(columns=self.DATASETS_COLUMNS)
        df = df.loc[(df[list(filters)] == pd.Series(filters)).all(axis=1)].copy()
        return df.sort_values(self.DATASETS_COLUMNS).dropna(subset=['task_type'])

    def get_dataset_id(self, dataset):
        datasets = self._load_table('datasets')
        dataset = datasets[datasets.dataset == dataset]
        if not dataset.empty:
            return dataset.iloc[0].dataset_id

    def get_best_pipeline(self, dataset):
        sdf = self._load_table('pipelines')
        dsdf = sdf[sdf.dataset == dataset]
        if dsdf.empty:
            dataset = self.get_dataset_id(dataset)

        dsdf = sdf[sdf.dataset == dataset]
        if not dsdf.empty:
            return dsdf.rename(columns={'_id': 'id'}).sort_values('rank').iloc[0]

    def load_pipeline(self, pipeline_id):
        return self._get_json('pipelines', pipeline_id)

    def load_template(self, template_name):
        return self._get_json('templates', template_name.replace('/', '.'))
