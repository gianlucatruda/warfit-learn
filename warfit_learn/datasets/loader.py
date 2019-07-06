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
from os.path import dirname, join


def load_iwpc() -> pd.DataFrame:
    """Loads raw IWPC dataset as dataframe object.

    Returns
    -------
    pd.DataFrame
        The raw IWPC dataset.
    """

    module_path = dirname(__file__)
    fpath = join(module_path, 'data', 'iwpc.pkl')

    df = pd.read_pickle(fpath)

    return df
