from collections import defaultdict

import os
from os import path

import pathlib

import pandas as pd


def concat_experiment_files(
    experiments: dict,
    files: list[str],
    dirs: list[str | pathlib.Path],
    is_key_path: bool = True,
) -> list[pd.DataFrame]:
    if len(files) != len(dirs):
        raise ValueError(f'Please provide equal number of files and directories, was {len(files)}--{len(dirs)}')

    dfs: defaultdict[str, list[pd.DataFrame]] = defaultdict(list)
    
    for exp_path, exp_name in experiments.items():
        if not is_key_path:
            exp_path, exp_name = exp_name, exp_path

        exp_path = str(exp_path)

        for d, f in zip(dirs, files):
            df = pd.read_csv(path.join(d, exp_path, f), sep=';', decimal=',')

            df['Experiment'] = exp_name

            dfs[f].append(df)

    return [pd.concat(df_list, ignore_index=True) for df_list in dfs.values()]


def get_subfolders(root) -> set[str]:
    paths = []

    for dirpath, dirnames, _ in os.walk(root):
        for dirname in dirnames:
            subfolder_path = path.relpath(path.join(dirpath, dirname), root)
            paths.append(subfolder_path)

    return set(paths)
