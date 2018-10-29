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

    def __init__(self, bucket, data_path='data'):
        self.bucket = bucket
        self.data_path = data_path
        self.csvs_path = os.path.join(data_path, 'csvs')

        if not os.path.exists(self.csvs_path):
            os.makedirs(self.csvs_path)

        self.dfs = dict()

    def download_csv(self, csv_name):
        LOGGER.info("Downloading %s csv from S3", csv_name)
        key = os.path.join('csvs', csv_name + '.csv.gz')

        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.bucket, Key=key)

        body_bytes = io.BytesIO(obj['Body'].read())
        gzip_file = gzip.GzipFile(fileobj=body_bytes, mode='rb')

        return pd.read_csv(gzip_file)

    def load_csv(self, csv_name):
        df = self.dfs.get(csv_name)
        if df is not None:
            return df

        pkl_filename = os.path.join(self.csvs_path, csv_name + '.pkl')
        if os.path.exists(pkl_filename):
            with open(pkl_filename, 'rb') as pkl_file:
                df = pickle.load(pkl_file)

        else:
            df = self.download_csv(csv_name)
            with open(pkl_filename, 'wb') as pkl_file:
                pickle.dump(df, pkl_file)

        self.dfs[csv_name] = df

        return df

    def get_tests(self, **filters):
        ddf = self.load_csv('datasets')
        tdf = self.load_csv('tests')
        tdf = tdf.merge(ddf, how='left', on='dataset')
        return tdf.loc[(tdf[list(filters)] == pd.Series(filters)).all(axis=1)].copy()

    def get_test_results(self, **filters):
        rdf = self.load_csv('test_results')
        rdf = rdf.loc[(rdf[list(filters)] == pd.Series(filters)).all(axis=1)]
        # tdf = self.get_tests(**filters)
        tdf = self.load_csv('tests')
        return rdf.merge(tdf, how='left', on=['dataset', 'test_id'], suffixes=('', '_results'))

    def get_pipelines(self, **filters):
        df = self.load_csv('solutions')
        df['pipeline'] = df['name']
        return df.loc[(df[list(filters)] == pd.Series(filters)).all(axis=1)].copy()

    def get_datasets(self, **filters):
        tests = self.get_tests(**filters)
        return tests[DATASETS_COLUMNS].sort_values(DATASETS_COLUMNS).drop_duplicates()

    def get_dataset_id(self, dataset):
        datasets = self.load_csv('datasets')
        dataset = datasets[datasets.dataset == dataset]
        if not dataset.empty:
            return dataset.iloc[0].dataset_id

    def get_json(self, folder, pipeline_id):
        key = os.path.join(folder, pipeline_id + '.json.gz')
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.bucket, Key=key)

        body_bytes = io.BytesIO(obj['Body'].read())
        gzip_file = gzip.GzipFile(fileobj=body_bytes, mode='rb')
        return json.load(gzip_file)

    def get_best_pipeline(self, dataset):
        sdf = self.load_csv('solutions')
        dsdf = sdf[sdf.dataset == dataset]
        if dsdf.empty:
            dataset = self.get_dataset_id(dataset)

        dsdf = sdf[sdf.dataset == dataset]
        if not dsdf.empty:
            return dsdf.sort_values('rank').iloc[0]

    def load_best_pipeline(self, dataset):
        pipeline = self.get_best_pipeline(dataset)
        return self.get_json('pipelines', pipeline._id)

    def load_template(self, template_id):
        return self.get_json('templates', template_id)
