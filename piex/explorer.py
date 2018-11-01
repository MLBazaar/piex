# -*- coding: utf-8 -*-

import gzip
import io
import json
import logging
import os
import pickle

import boto3
import pandas as pd

LOGGER = logging.getLogger(__name__)

DATASETS_COLUMNS = ['dataset', 'data_modality', 'task_type', 'task_subtype']


class PipelineExplorer:

    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.csvs_path = os.path.join(data_path, 'csvs')

        if not os.path.exists(self.csvs_path):
            os.makedirs(self.csvs_path)

        self.dfs = dict()

    def get_table(self, table_name):
        raise NotImplementedError

    def load_table(self, table_name):
        df = self.dfs.get(table_name)
        if df is not None:
            return df

        pkl_filename = os.path.join(self.csvs_path, table_name + '.pkl')
        if os.path.exists(pkl_filename):
            with open(pkl_filename, 'rb') as pkl_file:
                df = pickle.load(pkl_file)

        else:
            df = self.get_table(table_name)
            with open(pkl_filename, 'wb') as pkl_file:
                pickle.dump(df, pkl_file)

        self.dfs[table_name] = df

        return df

    def get_tests(self, **filters):
        ddf = self.load_table('datasets')
        tdf = self.load_table('tests')
        tdf = tdf.merge(ddf, how='left', on='dataset')
        return tdf.loc[(tdf[list(filters)] == pd.Series(filters)).all(axis=1)].copy()

    def get_test_results(self, **filters):
        rdf = self.load_table('test_results')
        rdf = rdf.loc[(rdf[list(filters)] == pd.Series(filters)).all(axis=1)]
        tdf = self.load_table('tests')
        return rdf.merge(tdf, how='left', on=['dataset', 'test_id'], suffixes=('', '_results'))

    def get_pipelines(self, **filters):
        df = self.load_table('solutions')
        df['pipeline'] = df['name']
        return df.loc[(df[list(filters)] == pd.Series(filters)).all(axis=1)].copy()

    def get_datasets(self, **filters):
        tests = self.get_tests(**filters)
        return tests[DATASETS_COLUMNS].sort_values(DATASETS_COLUMNS).drop_duplicates()

    def get_dataset_id(self, dataset):
        datasets = self.load_table('datasets')
        dataset = datasets[datasets.dataset == dataset]
        if not dataset.empty:
            return dataset.iloc[0].dataset_id

    def get_best_pipeline(self, dataset):
        sdf = self.load_table('solutions')
        dsdf = sdf[sdf.dataset == dataset]
        if dsdf.empty:
            dataset = self.get_dataset_id(dataset)

        dsdf = sdf[sdf.dataset == dataset]
        if not dsdf.empty:
            return dsdf.rename({'_id': 'id'}).sort_values('rank').iloc[0]

    def load_pipeline(self, pipeline_id):
        raise NotImplementedError

    def load_template(self, template_id):
        raise NotImplementedError

    def load_best_pipeline(self, dataset):
        pipeline = self.get_best_pipeline(dataset)
        return self.load_pipeline(pipeline._id)


class MongoPipelineExplorer(PipelineExplorer):

    def __init__(self, db, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = db

    def get_collection(self, collection, query=None, project=None):
        cursor = self.db[collection].find(query or dict(), project)
        return pd.DataFrame(list(cursor))

    def get_tests(self, **filters):
        ddf = self.get_collection('datasets', filters, {'_id': 0})
        datasets = list(ddf.dataset.unique())
        tdf = self.get_collection('tests', {'dataset': {'$in': datasets}}, {'_id': 0})
        tdf = tdf.merge(ddf, how='left', on='dataset')
        return tdf.loc[(tdf[list(filters)] == pd.Series(filters)).all(axis=1)].copy()

    def get_test_results(self, **filters):
        rdf = self.get_collection('test_results', filters, {'_id': 0})
        test_ids = list(rdf.test_id.unique())
        tdf = self.get_collection('tests', {'test_id': {'$in': test_ids}}, {'_id': 0})
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

        df = self.get_collection('solutions', filters, project)
        df['pipeline'] = df['name']
        loader = df.pop('loader')
        df['data_modality'] = loader.apply(lambda l: l.get('data_modality'))
        df['task_type'] = loader.apply(lambda l: l.get('task_type'))
        return df

    def get_datasets(self, **filters):
        tests = self.get_tests(**filters)
        return tests[DATASETS_COLUMNS].sort_values(DATASETS_COLUMNS).drop_duplicates()

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

    def load_pipeline(self, pipeline_id):
        return self.db.solutions.find_one({'_id': pipeline_id})

    def load_template(self, template_id):
        return self.db.templates.find_one({'_id': template_id})


class S3PipelineExplorer(PipelineExplorer):

    def __init__(self, bucket, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket = bucket

    def get_table(self, table_name):
        LOGGER.info("Downloading %s csv from S3", table_name)
        key = os.path.join('csvs', table_name + '.csv.gz')

        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.bucket, Key=key)

        body_bytes = io.BytesIO(obj['Body'].read())
        gzip_file = gzip.GzipFile(fileobj=body_bytes, mode='rb')

        return pd.read_csv(gzip_file)

    def get_json(self, folder, pipeline_id):
        key = os.path.join(folder, pipeline_id + '.json.gz')
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.bucket, Key=key)

        body_bytes = io.BytesIO(obj['Body'].read())
        gzip_file = gzip.GzipFile(fileobj=body_bytes, mode='rb')
        return json.load(gzip_file)

    def load_pipeline(self, pipeline_id):
        return self.get_json('pipelines', pipeline._id)

    def load_template(self, template_id):
        return self.get_json('templates', template_id)
