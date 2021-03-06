# Warfit-learn

[![PyPI version fury.io](https://badge.fury.io/py/warfit-learn.svg)](https://pypi.org/project/warfit-learn/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/warfit-learn.svg)](https://pypi.org/project/warfit-learn/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/467076ac86de4ca389febcf07bffd161)](https://app.codacy.com/app/gianlucatruda/warfit-learn?utm_source=github.com\&utm_medium=referral\&utm_content=gianlucatruda/warfit-learn\&utm_campaign=Badge_Grade_Dashboard)
[![PyPI license](https://img.shields.io/pypi/l/warfit-learn.svg)](https://pypi.org/project/warfit-learn/)

*A machine learning toolkit for reproducible research in warfarin dose estimation.*

Read [the paper on arXiv](https://arxiv.org/abs/1907.05363) for free or on [Science Direct](https://doi.org/10.1016/j.jbi.2020.103634) with your institutional access.

## Contents

*   [Features](#features)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Citing this work](#citing-this-work)

## Features

*   Seamless loading, cleaning, and preprocessing of the IWPC warfarin dataset.
*   Standardised implementations of scoring functions.
*   Percentage patients within 20% of therapeutic dose (PW20)
*   Mean absolute error (MAE)
*   R<sup>2</sup> coefficient
*   Hybrid scoring functions
*   Confidence intervals
*   Multithreaded model evaluation using standardised resampling techniques.
*   Monte-carlo cross validation
*   Bootstrap resampling
*   Full interoperability with NumPy, SciPy, Pandas, Scikit-learn, and MLxtend.

Supports Python 3.6+ on macOS, Linux, and Windows.

## Installation

```bash
pip install warfit-learn
```

## Usage

For a detailed tutorial, see the [Getting Started](https://github.com/gianlucatruda/warfit-learn/blob/master/docs/warfit_learn_tutorial.ipynb) document.

**Seamless loading and preprocessing of IWPC dataset**

```python
from warfit_learn import datasets, preprocessing
raw_iwpc = datasets.load_iwpc()
data = preprocessing.prepare_iwpc(raw_iwpc)
```

**Full scikit-learn interoperability**

```python
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from warfit_learn.estimators import Estimator
my_models = [
    Estimator(LinearRegression(), 'LR'),
    Estimator(LinearSVR(loss='epsilon_insensitive'), 'SVR'),
]
```

**Seamless, multithreaded research**

```python
from warfit_learn.evaluation import evaluate_estimators
results = evaluate_estimators(
    my_models,
    data,
    parallelism=0.5,
    resamples=10,
)
```

## Citing this work

If you use warfit-learn in a scientific publication, please consider citing the following paper:

G. Truda and P. Marais, Evaluating warfarin dosing models on multiple datasets with a novel software framework and evolutionary optimisation, *Journal of Biomedical Informatics* (2020), doi: https://doi.org/10.1016/j.jbi.2020.103634

BibTeX entry:

```bibtex
@article{Truda2020warfit,
  title = "Evaluating warfarin dosing models on multiple datasets with a novel software framework and evolutionary optimisation",
  journal = "Journal of Biomedical Informatics",
  pages = "103634",
  year = "2020",
  issn = "1532-0464",
  doi = "https://doi.org/10.1016/j.jbi.2020.103634",
  url = "http://www.sciencedirect.com/science/article/pii/S1532046420302628",
  author = "Gianluca Truda and Patrick Marais",
  keywords = "Warfarin, Machine learning, Genetic programming, Python, Supervised learning, Anticoagulant, Pharmacogenetics, Software",
}
```

## Copyright

Copyright (C) 2019 Gianluca Truda

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
