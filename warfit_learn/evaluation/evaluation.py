"""
A toolkit for reproducible research in warfarin dose estimation.
Copyright (C) 2019 Gianluca Truda

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import List
from tqdm import tqdm
from ..estimators import Estimator
from ..metrics import score_mae, score_pw20, score_r2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import resample
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from tabulate import tabulate


def evaluate_estimator(estimator: Estimator,
                       data: pd.DataFrame,
                       technique='mccv',
                       target_column='Therapeutic Dose of Warfarin',
                       resamples=100,
                       test_size=0.2,
                       squaring=False,
                       parallelism=0.5):
    """ Resampling evaluator for estimators using PW20, MAE, R^2.
    """

    assert(technique in ['mccv', 'bootstrap'])
    results = {'PW20': [], 'MAE': [], 'R2': []}

    avail_cores = multiprocessing.cpu_count()
    num_cores = max(int(avail_cores * parallelism), 1)

    try:
        if technique == 'bootstrap':
            replace = True
        elif technique == 'mccv':
            replace = False

        res = Parallel(n_jobs=num_cores)(delayed(_train_eval)(
            estimator,
            data,
            target_column=target_column,
            test_size=test_size,
            squaring=squaring,
            replace=replace) for i in range(resamples))

        results['PW20'] = [r[0] for r in res]
        results['MAE'] = [r[1] for r in res]
        results['R2'] = [r[2] for r in res]

        return results

    except Exception as e:
        print("Error occurred:", e)
        return results


def evaluate_estimators(estimators: List[Estimator],
                        data: pd.DataFrame,
                        target_column='Therapeutic Dose of Warfarin',
                        scale=True,
                        parallelism=0.5,
                        *args,
                        **kwargs):

    _data = data.copy()

    avail_cores = multiprocessing.cpu_count()
    num_cores = max(int(avail_cores * parallelism), 1)
    print(f'Using {num_cores} / {avail_cores} CPU cores...')

    if scale:
        x_cols = list(_data.columns)
        x_cols.remove(target_column)
        scaler = StandardScaler()
        _data[x_cols] = scaler.fit_transform(_data[x_cols])

    results = []
    for i in range(len(estimators)):
            est = estimators[i]
            print(f'\n{est.identifier}...')
            res = evaluate_estimator(est, _data,
                                     target_column=target_column,
                                     *args, **kwargs)
            res_dict = {
                'Estimator': [est.identifier for x in range(len(res['PW20']))],
                'PW20': res['PW20'],
                'MAE': res['MAE'],
                'R2': res['R2'],
            }
            prog = {k: [np.mean(res_dict[k])]
                    for k in list(res_dict.keys())[1:]}
            print(tabulate(prog, headers=prog.keys()))
            results.append(res_dict)

    # Compile results to single DF
    df_res = pd.DataFrame()
    for res in results:
        df_res = df_res.append(pd.DataFrame.from_dict(res))

    print(f"\n\n{df_res.groupby(['Estimator']).agg(np.mean)}\n")

    return df_res


def _train_eval(estimator: Estimator,
                data,
                test_size,
                target_column,
                squaring,
                replace=False):

    train, test = train_test_split(data, test_size=test_size)
    if replace:
        # Bootstrap resampling
        train = resample(train, replace=True)
    y_train = train[target_column].values
    x_train = train.drop([target_column], axis=1).values
    y_test = test[target_column].values
    # Square the dose (to undo upstream sqrt call)
    if squaring:
        y_test = np.square(y_test)
    x_test = test.drop([target_column], axis=1).values

    estimator.fit(x_train, y_train)
    predicts = estimator.predict(x_test)
    if squaring:
        predicts = np.square(predicts)

    return (
        score_pw20(y_test, predicts),
        score_mae(y_test, predicts),
        score_r2(y_test, predicts))
