"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from scipy import nanmean
import math


def load_unicef_data(fname="", data=None):
    """Loads Unicef data from JSON file.

    Retrieves a matrix of all rows and columns from rental listing data
    dataset.

    Args:
      none

    Returns:
      listing_id,feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N listing_ids
      features: vector of F feature names
      values: matrix N-by-F
    """
    if fname != "":
      # Uses pandas to help with string-NaN-numeric data.
       data = pd.read_json(fname)
    # find feature names.
    features = data.axes[1][:]
    features = features.tolist()
    listing_id = data.axes[0][1:]
    values = data.values[:, :]

    return (listing_id, features, values, data)



def not_digit_and_underline(a):
    digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_']
    if a in digit:
        return False
    else:
        return True
