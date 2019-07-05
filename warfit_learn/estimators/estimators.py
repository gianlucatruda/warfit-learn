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

from sklearn.base import BaseEstimator


class Estimator(BaseEstimator):
    """Wrapper class for any sklearn Estimator.
    """

    def __init__(self, estimator: BaseEstimator, identifier: str):
        if not isinstance(identifier, str):
            identifier = str(identifier)
        if not isinstance(estimator, BaseEstimator):
            raise TypeError
        self.__identifier = identifier
        self.__estimator = estimator
        self.__is_trained = False

    @property
    def identifier(self):
        """Get or set the identifier (name) for the Estimator.

        Returns
        -------
        str
            The identifier (name) for the Estimator.
        """
        return self.__identifier

    @property
    def estimator(self):
        return self.__estimator

    @property
    def is_trained(self):
        return self.__is_trained

    def fit(self, x, y):
        self.__estimator.fit(x, y)
        self.__is_trained = True

    def predict(self, x):
        return self.__estimator.predict(x)
