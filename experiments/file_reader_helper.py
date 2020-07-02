import json
import numpy as np
import pandas as pd

def get_experiments_from_fs(path):
    assert (path / '_resources/').exists() & (path / '_sources/').exists(), f"Bad path: {path}"
    exps = {}
    dfs = []

    for job in path.glob("*"):
        if job.parts[-1] in ['_resources', '_sources']:
            continue
        job_id = job.parts[-1]
        run = job / 'run.json'
        config = job / 'config.json'
        with open(run) as data_file:
            run = json.load(data_file)
        with open(config) as data_file:
            config = json.load(data_file)
        exps[job_id] = {**config, **run}

        metrics = job / 'metrics.json'
        with open(metrics) as data_file:
            metrics = json.load(data_file)
        if metrics:
            for metric, v in metrics.items():
                df = pd.DataFrame(v)
                df.index = pd.MultiIndex.from_product([[job_id], [metric], df.index], names=['_id', 'metric', 'index'])
                dfs += [df]

    exps = pd.DataFrame(exps).T
    pd.DataFrame(exps).index.name = '_id'
    df = pd.concat(dfs).drop('timestamps', axis=1)

    return exps, df


def process_dictionary_column(df, column_name):
    return df.drop(column_name, 1).assign(**pd.DataFrame(df[column_name].values.tolist(), index=df.index))


def process_tuple_column(df, column_name, output_column_names):
    return df.drop(column_name, 1).assign(**pd.DataFrame(df[column_name].values.tolist(), index=df.index))
