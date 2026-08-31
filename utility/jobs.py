import os

import pathlib

from utility.data import get_subfolders


def build_missing_ip_jobs(
    dir_exp: pathlib.Path,
    dir_mi: pathlib.Path,
    compute_mi: bool,
    exclude_tmps: bool,
    output: pathlib.Path,
):
    mi_folders = get_subfolders(dir_mi)
    exp_folders = get_subfolders(dir_exp)

    missing_folders = [m for m in list(exp_folders - mi_folders)]
    
    lines = []

    tmps = ['test', 'tmp', 'debug']

    for missing in sorted(missing_folders):
        if exclude_tmps and any(tmp in missing for tmp in tmps):
            continue

        p = dir_exp.joinpath(missing).as_posix()

        line = f'python main.py mi -s --plot-as-pdf --no-show-plots -d {p}'
        
        if not compute_mi:
            line = f'{line} --no-compute-mi'

        line += '\n'

        lines.append(line)

    create_script(output, lines)


def build_missing_ncs_jobs(
    experiment_groups: dict[str, dict[str, list]],
    dir_exp: pathlib.Path,
    dir_nc: pathlib.Path,
    n_epochs: int,
    exclude_tmps: bool,
    output: pathlib.Path,
):
    experiments = [os.path.relpath(experiment) for ds_data in experiment_groups.values()
                   for group_data in ds_data.values()
                   for experiment in group_data]

    exp_folders = get_subfolders(dir_exp)
    nc_folders = get_subfolders(dir_nc)

    missing_folders = [m for m in list(exp_folders - nc_folders) if m in experiments]

    lines = []

    tmps = ['test', 'tmp', 'debug']

    for missing in sorted(missing_folders):
        if exclude_tmps and any(tmp in missing for tmp in tmps):
            continue

        p = dir_exp.joinpath(missing).as_posix()

        line = f'python main.py nc compute --data {p} -n {n_epochs}\n'

        lines.append(line)

    create_script(output, lines)


def create_script(
    script_file: pathlib.Path,
    lines: list[str],
):
    os.makedirs(script_file.parent, exist_ok=True)

    with open(script_file, 'w') as f:
        f.write('#!/bin/bash\n')

        f.writelines(lines)
