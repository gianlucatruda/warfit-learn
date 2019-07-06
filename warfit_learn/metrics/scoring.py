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

from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import sem, t
import numpy as np


def score_pw20(y_true, y_pred):
    """Custom metric function for PW20"""
    patients_in_20 = 0
    for i in range(len(y_true)):
        if 0.8 * y_true[i] < y_pred[i] < 1.2 * y_true[i]:
            patients_in_20 += 1
    return float(100 * patients_in_20 / len(y_true))


def score_mae(y_true, y_pred):
    """Scoring metric for MAE using sklearn metric for mean_absolute_error"""
    return mean_absolute_error(y_true, y_pred)


def score_r2(y_true, y_pred):
    """Scoring metric using sklearn's r2 metric"""
    return r2_score(y_true, y_pred)


def score_hybrid(y_true, y_pred):
    """Custom metric function. A hybrid of MAE and PW20"""
    return score_pw20(y_true, y_pred) / (score_mae(y_true, y_pred) ** 2)


def confidence_interval(data, confidence=0.95, *args, **kwargs):
    """Calculates confidence interval start and end for some data.

    Assumes data is independent and follows a t-distribution.

    Parameters
    ----------
    data : array-like
        The 1D-array or list of values.
    confidence : float, optional
        The conidence level (inverse of alpha), by default 0.95

    Returns
    -------
    Tuple
        (interval_start, interval_end)
    """

    return t.interval(confidence,
                      data.shape[0] - 1,
                      loc=np.mean(data),
                      scale=sem(data))
