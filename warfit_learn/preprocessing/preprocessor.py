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

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

FILTER_COLUMNS = [
    'Age',
    'Therapeutic Dose of Warfarin',
    'Subject Reached Stable Dose of Warfarin',
    'Gender',
    'CYP2C9 consensus',
    'Imputed VKORC1',
]
RARE_ALLELES = [
    '*1/*5',
    '*1/*6',
    '*1/*11',
    '*1/*13',
    '*1/*14',
]

IWPC_PARAMS = [
    "Race (OMB)",
    "Age",
    "Height (cm)",
    "Weight (kg)",
    "Amiodarone (Cordarone)",
    "Carbamazepine (Tegretol)",
    "Phenytoin (Dilantin)",
    "Rifampin or Rifampicin",
    "Current Smoker",
    "CYP2C9 consensus",
    "Imputed VKORC1",
    "Therapeutic Dose of Warfarin",
    'INR on Reported Therapeutic Dose of Warfarin',
]


def prepare_iwpc(data: pd.DataFrame, drop_inr=True):
    """Prepare IWPC data for experimentation.

    NOTE: This is equivalent to calling `clean_iwpc()` and
    then `format_iwpc()`.

    Parameters
    ----------
    data : pd.DataFrame
        The raw IWPC data.
    drop_inr : bool, optional
        Whether to drop the INR field, by default True

    Returns
    -------
    pd.DataFrame
        The cleaned and preprocessed IWPC dataset.
    """

    assert(isinstance(data, pd.DataFrame))
    assert(_verify_shape(data))

    _data = clean_iwpc(data)
    _data = format_iwpc(_data)

    assert(_data.shape[0] == 5741)

    if drop_inr:
        _data.drop('INR on Reported Therapeutic Dose of Warfarin',
                   axis=1, inplace=True)

    return _data


def clean_iwpc(data: pd.DataFrame):
    """Clean the IWPC dataset.

    Imputes missing height and weight using linear regression.
    Imputes missing vkorc1 genotypes using IWPC algorithm.
    Vectorises categorical features to one-hot encoded format.

    NOTE: The output is not yet ready for training a model. You should
    call `format_iwpc(data)` with the output `data` to vectorise into
    an ML-ready format.

    Parameters
    ----------
    data : pd.DataFrame
        The raw IWPC dataset.

    Returns
    -------
    pd.DataFrame
        The cleaned IWPC dataset.
    """

    assert(isinstance(data, pd.DataFrame))
    assert(_verify_shape(data))

    _data = data.copy()
    _data = _drop_height_weight(_data)
    _data = _drop_race_gender(_data)
    _data = _define_dose_groups(_data)
    _dummies = _get_dummy_categoricals(_data)
    _heights = _get_imputed_heights(_dummies)
    _weights = _get_imputed_weights(_dummies)
    _data = _replace_height_and_weight(_data, _heights, _weights)
    _data = _impute_genotypes(_data)
    _data = _drop_unusable_rows(_data)
    _data = _exclude_rare_alleles(_data)
    _data = _exclude_extreme_doses(_data)

    return _data


def format_iwpc(data: pd.DataFrame, mode='df', params=IWPC_PARAMS):
    """Format cleaned IWPC dataset into ML-ready dataframe.

    NOTE: This requires a cleaned IWPC dataset, i.e. the output of the
    `clean_iwpc()` function.

    Parameters
    ----------
    data : pd.DataFrame
        The cleaned IWPC dataset.
    mode : str, optional
        The return mode, either 'df' for a Pandas dataframe or 'array'
        for two NumPy arrays, by default 'df'.
    params : list-like, optional
        List of the which parameters should be included in the output,
        by default `IWPC_PARAMS` (the ones standardised in the research).

    Returns
    -------

    pd.DataFrame
        A dataframe of the formatted IWPC data, if in `df` mode.

    Tuple
        A tuple of (X, y) numpy arrays where `X` is the multi-dimensional
        input matrix and `y` is the single-dimensional target feature.
        This is only returned if `array` mode.

    Raises
    ------
    KeyError
        All parameters in `params` must be the name of a column in `data`.
    """

    assert(isinstance(data, pd.DataFrame))

    for p in params:
        if p not in data.columns:
            raise KeyError(f"key '{p}' is not in the data provided")

    _data = data.copy()

    _data = pd.get_dummies(_data[params])

    # Fill NaNs with zero
    _data = _data.fillna(value=0)

    # Ensure that no NaNs remain
    assert(_data.isnull().values.any() == False)

    if mode == 'array':
        y = _data['Therapeutic Dose of Warfarin'].values
        X = _data.drop(['Therapeutic Dose of Warfarin'], axis=1).values
        return X, y
    else:
        return _data


def describe_iwpc_cohort(data: pd.DataFrame):
    """Describes the distribution of the cohort.

    Parameters
    ----------
    data : pd.DataFrame
        The IWPC dataset after `clean_iwpc()` has been run on it.
    """

    _interest_columns = [
        'Therapeutic Dose of Warfarin',
        'Height (cm)',
        'Weight (kg)',
        'INR on Reported Therapeutic Dose of Warfarin',
    ]

    _meds_of_interest = [
        'Carbamazepine (Tegretol)',
        'Phenytoin (Dilantin)',
        'Rifampin or Rifampicin',
        'Amiodarone (Cordarone)',
    ]

    _enzyme_inducers = [
        'Carbamazepine (Tegretol)',
        'Phenytoin (Dilantin)',
        'Rifampin or Rifampicin',
    ]

    for i in _interest_columns:
        print(data[i].describe(), end='\n\n')

    for race in ['Asian', 'White', 'Black']:
        print(data['Race (OMB)'][data['Race (OMB)'].str.contains(
            race)].describe(), end='\n\n')

    print(data['Age'].value_counts().sort_values(), end='\n\n')

    for med in _meds_of_interest:
        print(data[med][data[med] == 1].value_counts(), end='\n\n')

    print(
        'Patients on enzyme enducers: ',
        data[_enzyme_inducers][data[_enzyme_inducers].any(axis=1)].shape[0],
        end='\n\n'
    )

    print(data['Imputed VKORC1'].value_counts().sort_values(), end='\n\n')

    print(data['CYP2C9 consensus'].value_counts(
    ).sort_values(ascending=False), end='\n\n')

    for cat in ['Gender', 'Current Smoker']:
        print(data[cat].value_counts(), end='\n\n')


def _verify_shape(df: pd.DataFrame):
    """Ensures data it of specific shape
    """

    return df.shape == (6256, 68)


def _drop_height_weight(df: pd.DataFrame):
    """Drop rows with both height AND weight missing
    """

    _df = df.copy()
    _df.dropna(subset=['Weight (kg)', 'Height (cm)'], how='all', inplace=True)

    return _df


def _drop_race_gender(df: pd.DataFrame):
    """Drop rows where race AND gender are missing
    """

    _df = df.copy()
    _df.dropna(subset=['Race (OMB)', 'Gender'], inplace=True)

    return _df


def _define_dose_groups(df: pd.DataFrame):
    """Classify dose (low, inter, high) based on race and dose distribution
    """

    RACE = 'Race (OMB)'
    DOSE = 'Therapeutic Dose of Warfarin'
    GROUP = 'dose_group'

    # df.loc[DOSE] = df[DOSE].astype(float)

    def _group(x, lower, upper):
        if x[DOSE] < lower:
            return 'low'
        elif x[DOSE] > upper:
            return 'high'
        else:
            return 'inter'

    df[GROUP] = 'none'
    for race in df[RACE].unique():
        dist = df[df[RACE] == race][DOSE].describe()
        df.loc[df[RACE] == race, GROUP] = df[df[RACE] == race].apply(
            lambda x: _group(x, float(dist['25%']), float(dist['75%'])), axis=1)

    return df


def _get_dummy_categoricals(df: pd.DataFrame):
    """One-hot encode categorical columns
    """

    _dummied = pd.get_dummies(
        df[[
            'Weight (kg)',
            'Height (cm)',
            'Race (OMB)',
            'Gender']],
        columns=['Race (OMB)', 'Gender'])

    return _dummied


def _get_imputed_heights(df: pd.DataFrame):
    """Impute height using weight, race, sex
    """

    train = df[(df['Height (cm)'].isnull() == False) & (
        df['Weight (kg)'].isnull() == False)]
    pred = df[(df['Height (cm)'].isnull())]

    x_train = train.drop(['Height (cm)'], axis='columns')
    y_train = train['Height (cm)']
    x_pred = pred.drop(['Height (cm)'], axis='columns')
    y_pred = pred['Height (cm)']
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    imputed_heights = linreg.predict(x_pred)

    return imputed_heights


def _get_imputed_weights(df: pd.DataFrame):
    """Impute weight using height, race, sex
    """

    train = df[(df['Weight (kg)'].isnull() == False) & (
        df['Height (cm)'].isnull() == False)]
    pred = df[(df['Weight (kg)'].isnull())]

    x_train = train.drop(['Weight (kg)'], axis='columns')
    y_train = train['Weight (kg)']
    x_pred = pred.drop(['Weight (kg)'], axis='columns')
    y_pred = pred['Weight (kg)']

    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    imputed_weights = linreg.predict(x_pred)

    return imputed_weights


def _replace_height_and_weight(data: pd.DataFrame,
                               heights: np.array,
                               weights: np.array,
                               ):
    """Insert imputed heights and weights into data
    """

    _data = data.copy()

    _data.loc[_data['Height (cm)'].isnull(), 'Height (cm)'] = heights
    _data.loc[_data['Weight (kg)'].isnull(), 'Weight (kg)'] = weights
    # Sanity check
    assert(_data[['Height (cm)', 'Weight (kg)']
                 ].isnull().values.any() == False)

    return _data


def _impute_vkorc1_row(row: pd.Series):
    """Impute VKORCI genotype using Klein et al. 2009 technique
    """

    rs2359612 = row['VKORC1 genotype:   2255C>T (7566); chr16:31011297; rs2359612; A/G']
    rs9934438 = row['VKORC1 genotype:   1173 C>T(6484); chr16:31012379; rs9934438; A/G']
    rs9923231 = row['VKORC1 genotype:   -1639 G>A (3673); chr16:31015190; rs9923231; C/T']
    rs8050894 = row['VKORC1 genotype:   1542G>C (6853); chr16:31012010; rs8050894; C/G']
    race = row['Race (OMB)']
    black_missing_mixed = [
        'Black or African American',
        'Missing or Mixed Race']

    if rs9923231 in ['A/A', 'A/G', 'G/A', 'G/G']:
        return rs9923231
    elif race not in black_missing_mixed and rs2359612 == 'C/C':
        return 'G/G'
    elif race not in black_missing_mixed and rs2359612 == 'T/T':
        return 'A/A'
    elif rs9934438 == 'C/C':
        return 'G/G'
    elif rs9934438 == 'T/T':
        return 'A/A'
    elif rs9934438 == 'C/T':
        return 'A/G'
    elif race not in black_missing_mixed and rs8050894 == 'G/G':
        return 'G/G'
    elif race not in black_missing_mixed and rs8050894 == 'C/C':
        return 'A/A'
    elif race not in black_missing_mixed and rs8050894 == 'C/G':
        return 'A/G'
    else:
        return 'Unknown'


def _impute_genotypes(df: pd.DataFrame, func=_impute_vkorc1_row):
    """Impute and insert VKORC1 and CYP2C9 for each row
    """

    _df = df.copy()

    # Impute VKORC1 genotypes
    _df['Imputed VKORC1'] = _df.apply(func, axis=1)

    # Convert NaN CYP2C9 genotypes to 'Missing'
    _df.loc[_df['CYP2C9 consensus'].isna(), 'CYP2C9 consensus'] = 'Unknown'

    return _df


def _drop_unusable_rows(df: pd.DataFrame, col_names=FILTER_COLUMNS):
    """Remove essential rows that are missing
    """

    _df = df.copy()
    _df.dropna(subset=col_names, inplace=True)

    return _df


def _exclude_rare_alleles(df: pd.DataFrame, alleles=RARE_ALLELES):
    """Remove rows with rare alleles
    """

    _df = df[df['CYP2C9 consensus'].isin(alleles) == False]
    return _df


def _exclude_extreme_doses(df: pd.DataFrame, threshold=315):
    """Exclude extreme weekly warfarin doses
    """

    _df = df[df['Therapeutic Dose of Warfarin'] < 315]
    return _df
