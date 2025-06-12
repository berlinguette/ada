from typing import Tuple, Optional, Iterator, Union
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import warnings
import operator
import shutil
import psutil
import os


def get_cpu_count() -> int:
    """
    Determine the number of physical cores.
    :return: Number of physical cores.
    """
    return psutil.cpu_count(logical=False)


def calc_rmse(
        y0: np.ndarray,
        y1: np.ndarray,
) -> np.array:
    """
    Calculate the RMSE.
    https://en.wikipedia.org/wiki/Root-mean-square_deviation
    :param y0: of shape n_samples x n_ydim
    :param y1: of shape n_samples x n_ydim
    :return: RMSE.
    """

    assert y0.shape == y1.shape
    return np.sqrt(np.sum((y0 - y1) ** 2, axis=0)/y0.shape[0])


def dict_to_json_safe(
        d: dict,
) -> dict:
    """
    Convert a dictionary to a form that can be saved to saved to json
    :param d: dictionary to convert
    :return: dictionary to save
    """

    ds = dict()

    for key, value in d.items():
        if type(value) in [str, int, float]:
            ds[key] = value
        elif type(value) is np.float_:
            ds[key] = float(value)
        elif type(value) is np.int_:
            ds[key] = int(value)
        elif type(value) == np.ndarray:
            ds[key] = str(value)
        elif value in [True, False, None]:
            ds[key] = str(value)
        elif type(value) is dict:
            ds[key] = dict_to_json_safe(value)
        elif hasattr(value, '__name__'):
            ds[key] = value.__name__
        elif key == 'model':
            ds[key] = value.__class__.__name__

        else:
            raise ValueError('Unable to identify JSON data type.')
    return ds


def generate_timestamp():
    """
    Generate a timestamp in the format YYYY-MM-DD_HH-MM-SS
    :return: string
    """
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def calc_enhancement(
        a: np.ndarray,
        b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    https://pubs.rsc.org.sci-hub.do/en/content/articlelanding/2020/sc/c9sc05999g
    :param a: A 2D of size n runs x m length. Fast array.
    :param b: A 2D of size p runs x m length. Slow array.
    :return: Accelerations, geometric mean, geometric std
    """

    # Crop the arrays to the shortest length between the two
    min_length = min(a.shape[1], b.shape[1])
    a = a[:, :min_length]
    b = b[:, :min_length]

    # Create a place to store the enhancements
    enhance = np.zeros((a.shape[0] * b.shape[0], a.shape[1]))

    # Suppress div by 0 warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)

        # For each a row, b row
        for i, ii in enumerate(a):
            for j, jj in enumerate(b):

                # Place to store
                ij = (b.shape[0] * i) + j

                # Store
                enhance[ij] = ii / jj

    # If enhancement has a value of nan, this means there was a 0 / 0. This
    # should actually read as 1.
    enhance[np.isnan(enhance)] = 1

    # If enhancement has a value of inf, this means x / 0. Set these to nan.
    enhance[enhance == np.inf] = np.nan

    # Calculate geo mean, std
    mean = geometric_mean(enhance)
    std = geometric_std(enhance)

    # Return
    return enhance, mean, std


def geometric_mean(
        a: np.ndarray,
) -> np.ndarray:
    """
    Calculate the geometric mean.
    https://en.wikipedia.org/wiki/Geometric_mean
    Note that the traditional mechanism for calculating is not used
    :param a: array to compute
    :return: Geometric mean
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.exp(np.nanmean(np.log(a), axis=0))


def geometric_std(
        a: np.ndarray,
) -> np.ndarray:
    """
    Calculate the geometric std.
    https://en.wikipedia.org/wiki/Geometric_standard_deviation
    :param a: array to compute
    :return: geometric std
    """

    # Calculate the geometric mean
    mean = geometric_mean(a)

    # Calculate the geometric standard deviation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.exp((np.nansum(np.log(a / mean) ** 2, axis=0) /
                       np.sum(~np.isnan(a), axis=0)) ** 0.5)


def calc_acceleration(
        a: np.ndarray,
        b: np.ndarray,
        multiprocessing: bool = True,
        return_est_frac: bool = False,
) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
     ]:
    """
    Calculate the rate at which one array achieves a certain value faster than
    another. In practice, this is used to determine the acceleration of one
    optimization run over another. Two arrays are passed, a and b. The
    acceleration of a is calculated with respect to b. for example, if it takes
    a 8 samples to observe a value of 10, and it takes b 16 samples to observe
    a value of 10, then the acceleration of a with respect to b is 16/8 = 2.

    If, however, the value of b is greater than the value of a at a given point,
    then the acceleration of a w.r.t. b at that point is equal to the inverse of
    the acceleration of b w.r.t. a at that point. This logic gives symmetry to
    this acceleration calculation, such that the acceleration of a w.r.t. a is
    1.

    Acceleration was first described by Gregoire in the following publication:
    https://pubs.rsc.org/en/content/articlelanding/2020/sc/c9sc05999g
    (See top right of fourth page, second para, lower half.)

    We further describe acceleration in the methods section of this paper:
    https://bit.ly/3hFoeiK

    Note that the running best must be passed, not just the observed values. See
    calc_running_best() for more information. Both a and b must have the same
    n_length, or a.shape[1] == b.shape[1], but they do not need to have the
    same number of n_repeats (a.shape[0], b.shape[0].

    :param a: A 2D of size n_repeats x n_length. Assumed to be the fast array.
    :param b: A 2D of size n_repeats x n_length. Assumed to be the slow array.
    :param multiprocessing: Should multiprocessing be used.
    :return: accelerations, geometric mean, geometric std
    """

    # Define some frequently used constants
    n_repeats_a = a.shape[0]
    n_repeats_b = b.shape[0]
    n_length_a = a.shape[1]
    n_length_b = b.shape[1]

    # Define the shorter of the two lengths, and create cropped versions
    short_length = min(n_length_a, n_length_b)
    ac = a[:, :short_length]
    bc = b[:, :short_length]

    # Create a place to store the accelerations.
    indexes = np.zeros((n_repeats_a * n_repeats_b, short_length))

    # If multiprocessing
    if multiprocessing:

        # Setup multiprocessing objects
        pool = mp.Pool()
        results = list()

        # Iterate through both the runs in a and b.
        for ii, i in enumerate(a):
            for jj, j in enumerate(b):

                # Fill the pool with tasks
                results.append(pool.apply_async(
                    _calc_acceleration,
                    kwds=dict(
                        a=i,
                        b=j,
                    )
                ))

        # Complete the calculation
        for result in tqdm(results):
            result.get()
        pool.close()
        pool.join()

        # Store the data in the index arrays
        count = 0
        for i in range(len(a)):
            for j in range(len(b)):
                indexes[count] = results[count].get()
                count += 1

    # Else if performing serially
    else:

        # Create progress bar to visualize progress.
        pbar = tqdm(total=len(a) * len(b))

        # Iterate through both the runs in a and b.
        for ii, i in enumerate(a):
            for jj, j in enumerate(b):

                # Calculate the acceleration of a w.r.t. b, and b w.r.t. a.
                ij = (n_repeats_b * ii) + jj
                indexes[ij] = _calc_acceleration(i, j)

                # Update pbar
                pbar.update(1)

        # Close the progress bar
        pbar.close()

    # Create an array of controls to compare to. For example, if the calculated
    # index above is equal to this index, the acceleration is 1.
    control = np.arange(short_length)[None, :]

    # Calculate accelerations against the control. One is added, such that the
    # value is representative of the number of samples completed, and not the
    # sample index.
    accel = (indexes + 1) / (control + 1)

    # If the acceleration is nan, it means that the slow run never beat
    # the fast run. Therefore, the minimum possible acceleration can be found
    # assuming that the next unobserved value would beat the fast run. Replace
    # the nan values with this minimum acceleration value. First calculate the
    # minimum possible acceleration if no acceleration could be calculated.
    min_accel = (n_length_b + 1) / \
        np.repeat(
            np.arange(short_length)[None, :] + 1, (n_repeats_a * n_repeats_b),
            axis=0,
        )

    # If no acceleration could be calculated, replace it with this minimum
    # acceleration value.
    accel_est = np.where(
        np.isnan(accel),
        min_accel,
        accel,
    )

    # # The geometric mean of the accelerations is calculated. For example, the
    # # geometric mean of 0.25 and 4 is 1. Or, the geometric mean of a run that is
    # # four times slower and a run that is four times faster is an acceleration
    # # of one.
    mean = geometric_mean(accel_est)
    std = geometric_std(accel_est)

    # If the estimated fraction should be returned, calculate it.
    if return_est_frac:

        # Determine the fraction of runs that are lower-bound estimates
        est_frac = np.sum(np.isnan(accel), axis=0) / (n_repeats_a * n_repeats_b)

        # Return w/ est_frac
        return accel_est, mean, std, est_frac

    # Otherwise just return the accel, mean, std
    else:
        return accel_est, mean, std


def _calc_acceleration(
        a: np.ndarray,
        b: np.ndarray,
) -> np.ndarray:
    """
    This is a helper function for calc_acceleration. See calc_acceleration() for
    full documentation.
    :param a: A single run of length n_length. Assumed to be the fast array.
    :param b: A single run of length n_length. Assumed to be the slow array.
    :return:
    """

    # Crop the fast array to the length of the slow array.
    a = a[:b.shape[0]]

    # Identify where the fast run (a) is less than or equal to the slow run (b)
    # by performing a cross, and marking these values as True.
    cross = a <= b[:, None]

    # Identify the index where the fast run first is equal to or faster than the
    # slow run.
    index = np.argmax(cross, axis=0).astype(np.float64)

    # Identify where the fast run never beats the slow run, and mark it as nan.
    nanmask = np.all(~cross, axis=0)
    index[nanmask] = np.nan

    # Return
    return index


def calc_running_best(
        x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the running best sample
    :param x: An array of size n_repeats x n_length
    :return: An mean, std as array of size n_length
    """

    # Calculate the running max
    maxs = np.maximum.accumulate(x, axis=1)

    # Calculate mean, std
    mean = maxs.mean(axis=0)
    std = maxs.std(axis=0)

    # Return
    return maxs, mean, std


@dataclass
class ColorTemplate:
    i50: str
    i100: str
    i200: str
    i300: str
    i400: str
    i500: str
    i600: str
    i700: str
    i800: str
    i900: str
    iA100: str
    iA200: str
    iA400: str
    iA700: str


class _ColorRed(ColorTemplate):
    def __init__(self):
        self.i50 = '#FFEBEE'
        self.i100 = '#FFCDD2'
        self.i200 = '#EF9A9A'
        self.i300 = '#E57373'
        self.i400 = '#EF5350'
        self.i500 = '#F44336'
        self.i600 = '#E53935'
        self.i700 = '#D32F2F'
        self.i800 = '#C62828'
        self.i900 = '#B71C1C'
        self.iA100 = '#FF8A80'
        self.iA200 = '#FF5252'
        self.iA400 = '#FF1744'
        self.iA700 = '#D50000'


class _ColorPink:
    def __init__(self):
        self.i50 = '#FCE4EC'
        self.i100 = '#F8BBD0'
        self.i200 = '#F48FB1'
        self.i300 = '#F06292'
        self.i400 = '#EC407A'
        self.i500 = '#E91E63'
        self.i600 = '#D81B60'
        self.i700 = '#C2185B'
        self.i800 = '#AD1457'
        self.i900 = '#880E4F'
        self.iA100 = '#FF80AB'
        self.iA200 = '#FF4081'
        self.iA400 = '#F50057'
        self.iA700 = '#C51162'


class _ColorPurple:
    def __init__(self):
        self.i50 = '#F3E5F5'
        self.i100 = '#E1BEE7'
        self.i200 = '#CE93D8'
        self.i300 = '#BA68C8'
        self.i400 = '#AB47BC'
        self.i500 = '#9C27B0'
        self.i600 = '#8E24AA'
        self.i700 = '#7B1FA2'
        self.i800 = '#6A1B9A'
        self.i900 = '#4A148C'
        self.iA100 = '#EA80FC'
        self.iA200 = '#E040FB'
        self.iA400 = '#D500F9'
        self.iA700 = '#AA00FF'


class _ColorDeepPurple:
    def __init__(self):
        self.i50 = '#EDE7F6'
        self.i100 = '#D1C4E9'
        self.i200 = '#B39DDB'
        self.i300 = '#9575CD'
        self.i400 = '#7E57C2'
        self.i500 = '#673AB7'
        self.i600 = '#5E35B1'
        self.i700 = '#512DA8'
        self.i800 = '#4527A0'
        self.i900 = '#311B92'
        self.iA100 = '#B388FF'
        self.iA200 = '#7C4DFF'
        self.iA400 = '#651FFF'
        self.iA700 = '#6200EA'


class _ColorIndigo:
    def __init__(self):
        self.i50 = '#E8EAF6'
        self.i100 = '#C5CAE9'
        self.i200 = '#9FA8DA'
        self.i300 = '#7986CB'
        self.i400 = '#5C6BC0'
        self.i500 = '#3F51B5'
        self.i600 = '#3949AB'
        self.i700 = '#303F9F'
        self.i800 = '#283593'
        self.i900 = '#1A237E'
        self.iA100 = '#8C9EFF'
        self.iA200 = '#536DFE'
        self.iA400 = '#3D5AFE'
        self.iA700 = '#304FFE'


class _ColorBlue:
    def __init__(self):
        self.i50 = '#E3F2FD'
        self.i100 = '#BBDEFB'
        self.i200 = '#90CAF9'
        self.i300 = '#64B5F6'
        self.i400 = '#42A5F5'
        self.i500 = '#2196F3'
        self.i600 = '#1E88E5'
        self.i700 = '#1976D2'
        self.i800 = '#1565C0'
        self.i900 = '#0D47A1'
        self.iA100 = '#82B1FF'
        self.iA200 = '#448AFF'
        self.iA400 = '#2979FF'
        self.iA700 = '#2962FF'


class _ColorLightBlue:
    def __init__(self):
        self.i50 = '#E1F5FE'
        self.i100 = '#B3E5FC'
        self.i200 = '#81D4FA'
        self.i300 = '#4FC3F7'
        self.i400 = '#29B6F6'
        self.i500 = '#03A9F4'
        self.i600 = '#039BE5'
        self.i700 = '#0288D1'
        self.i800 = '#0277BD'
        self.i900 = '#01579B'
        self.iA100 = '#80D8FF'
        self.iA200 = '#40C4FF'
        self.iA400 = '#00B0FF'
        self.iA700 = '#0091EA'


class _ColorCyan:
    def __init__(self):
        self.i50 = '#E0F7FA'
        self.i100 = '#B2EBF2'
        self.i200 = '#80DEEA'
        self.i300 = '#4DD0E1'
        self.i400 = '#26C6DA'
        self.i500 = '#00BCD4'
        self.i600 = '#00ACC1'
        self.i700 = '#0097A7'
        self.i800 = '#00838F'
        self.i900 = '#006064'
        self.iA100 = '#84FFFF'
        self.iA200 = '#18FFFF'
        self.iA400 = '#00E5FF'
        self.iA700 = '#00B8D4'


class _ColorTeal:
    def __init__(self):
        self.i50 = '#E0F2F1'
        self.i100 = '#B2DFDB'
        self.i200 = '#80CBC4'
        self.i300 = '#4DB6AC'
        self.i400 = '#26A69A'
        self.i500 = '#009688'
        self.i600 = '#00897B'
        self.i700 = '#00796B'
        self.i800 = '#00695C'
        self.i900 = '#004D40'
        self.iA100 = '#A7FFEB'
        self.iA200 = '#64FFDA'
        self.iA400 = '#1DE9B6'
        self.iA700 = '#00BFA5'


class _ColorGreen:
    def __init__(self):
        self.i50 = '#E8F5E9'
        self.i100 = '#C8E6C9'
        self.i200 = '#A5D6A7'
        self.i300 = '#81C784'
        self.i400 = '#66BB6A'
        self.i500 = '#4CAF50'
        self.i600 = '#43A047'
        self.i700 = '#388E3C'
        self.i800 = '#2E7D32'
        self.i900 = '#1B5E20'
        self.iA100 = '#B9F6CA'
        self.iA200 = '#69F0AE'
        self.iA400 = '#00E676'
        self.iA700 = '#00C853'


class _ColorLightGreen:
    def __init__(self):
        self.i50 = '#F1F8E9'
        self.i100 = '#DCEDC8'
        self.i200 = '#C5E1A5'
        self.i300 = '#AED581'
        self.i400 = '#9CCC65'
        self.i500 = '#8BC34A'
        self.i600 = '#7CB342'
        self.i700 = '#689F38'
        self.i800 = '#558B2F'
        self.i900 = '#33691E'
        self.iA100 = '#CCFF90'
        self.iA200 = '#B2FF59'
        self.iA400 = '#76FF03'
        self.iA700 = '#64DD17'


class _ColorLime:
    def __init__(self):
        self.i50 = '#F9FBE7'
        self.i100 = '#F0F4C3'
        self.i200 = '#E6EE9C'
        self.i300 = '#DCE775'
        self.i400 = '#D4E157'
        self.i500 = '#CDDC39'
        self.i600 = '#C0CA33'
        self.i700 = '#AFB42B'
        self.i800 = '#9E9D24'
        self.i900 = '#827717'
        self.iA100 = '#F4FF81'
        self.iA200 = '#EEFF41'
        self.iA400 = '#C6FF00'
        self.iA700 = '#AEEA00'


class _ColorYellow:
    def __init__(self):
        self.i50 = '#FFFDE7'
        self.i100 = '#FFF9C4'
        self.i200 = '#FFF59D'
        self.i300 = '#FFF176'
        self.i400 = '#FFEE58'
        self.i500 = '#FFEB3B'
        self.i600 = '#FDD835'
        self.i700 = '#FBC02D'
        self.i800 = '#F9A825'
        self.i900 = '#F57F17'
        self.iA100 = '#FFFF8D'
        self.iA200 = '#FFFF00'
        self.iA400 = '#FFEA00'
        self.iA700 = '#FFD600'


class _ColorAmber:
    def __init__(self):
        self.i50 = '#FFF8E1'
        self.i100 = '#FFECB3'
        self.i200 = '#FFE082'
        self.i300 = '#FFD54F'
        self.i400 = '#FFCA28'
        self.i500 = '#FFC107'
        self.i600 = '#FFB300'
        self.i700 = '#FFA000'
        self.i800 = '#FF8F00'
        self.i900 = '#FF6F00'
        self.iA100 = '#FFE57F'
        self.iA200 = '#FFD740'
        self.iA400 = '#FFC400'
        self.iA700 = '#FFAB00'


class _ColorOrange:
    def __init__(self):
        self.i50 = '#FFF3E0'
        self.i100 = '#FFE0B2'
        self.i200 = '#FFCC80'
        self.i300 = '#FFB74D'
        self.i400 = '#FFA726'
        self.i500 = '#FF9800'
        self.i600 = '#FB8C00'
        self.i700 = '#F57C00'
        self.i800 = '#EF6C00'
        self.i900 = '#E65100'
        self.iA100 = '#FFD180'
        self.iA200 = '#FFAB40'
        self.iA400 = '#FF9100'
        self.iA700 = '#FF6D00'


class _ColorDeepOrange:
    def __init__(self):
        self.i50 = '#FBE9E7'
        self.i100 = '#FFCCBC'
        self.i200 = '#FFAB91'
        self.i300 = '#FF8A65'
        self.i400 = '#FF7043'
        self.i500 = '#FF5722'
        self.i600 = '#F4511E'
        self.i700 = '#E64A19'
        self.i800 = '#D84315'
        self.i900 = '#BF360C'
        self.iA100 = '#FF9E80'
        self.iA200 = '#FF6E40'
        self.iA400 = '#FF3D00'
        self.iA700 = '#DD2C00'


class _ColorBrown:
    def __init__(self):
        self.i50 = '#EFEBE9'
        self.i100 = '#D7CCC8'
        self.i200 = '#BCAAA4'
        self.i300 = '#A1887F'
        self.i400 = '#8D6E63'
        self.i500 = '#795548'
        self.i600 = '#6D4C41'
        self.i700 = '#5D4037'
        self.i800 = '#4E342E'
        self.i900 = '#3E2723'


class _ColorGray:
    def __init__(self):
        self.i50 = '#FAFAFA'
        self.i100 = '#F5F5F5'
        self.i200 = '#EEEEEE'
        self.i300 = '#E0E0E0'
        self.i400 = '#BDBDBD'
        self.i500 = '#9E9E9E'
        self.i600 = '#757575'
        self.i700 = '#616161'
        self.i800 = '#424242'
        self.i900 = '#212121'


class _ColorBlueGray:
    def __init__(self):
        self.i50 = '#ECEFF1'
        self.i100 = '#CFD8DC'
        self.i200 = '#B0BEC5'
        self.i300 = '#90A4AE'
        self.i400 = '#78909C'
        self.i500 = '#607D8B'
        self.i600 = '#546E7A'
        self.i700 = '#455A64'
        self.i800 = '#37474F'
        self.i900 = '#263238'


class _Colors:
    def __init__(self):
        # Full names
        self.red = _ColorRed()
        self.pink = _ColorPink()
        self.purple = _ColorPurple()
        self.deeppurple = _ColorDeepPurple()
        self.indigo = _ColorIndigo()
        self.blue = _ColorBlue()
        self.lightblue = _ColorLightBlue()
        self.cyan = _ColorCyan()
        self.teal = _ColorTeal()
        self.green = _ColorGreen()
        self.lightgreen = _ColorLightGreen()
        self.lime = _ColorLime()
        self.yellow = _ColorYellow()
        self.amber = _ColorAmber()
        self.orange = _ColorOrange()
        self.deeporange = _ColorDeepOrange()
        self.brown = _ColorBrown()
        self.gray = _ColorGray()
        self.bluegray = _ColorBlueGray()

        # Short names
        self.r = _ColorRed()
        self.p = _ColorPink()
        self.pu = _ColorPurple()
        self.dp = _ColorDeepPurple()
        self.i = _ColorIndigo()
        self.b = _ColorBlue()
        self.lb = _ColorLightBlue()
        self.c = _ColorCyan()
        self.t = _ColorTeal()
        self.g = _ColorGreen()
        self.lg = _ColorLightGreen()
        self.l = _ColorLime()
        self.y = _ColorYellow()
        self.a = _ColorAmber()
        self.o = _ColorOrange()
        self.do = _ColorDeepOrange()
        self.br = _ColorBrown()
        self.gr = _ColorGray()
        self.bg = _ColorBlueGray()


c = _Colors()


def colorset(
        pos: int,
        length: int,
        colorscale: str = 'viridis',
) -> Tuple[float]:
    """
    Generate a well-spaced color scale.
    :param pos: Position of color.
    :param length: Total length
    :param colorscale: matplotlib colorscale to use
    :return: list of colors
    """

    return plt.get_cmap(colorscale)(pos / length)


def _clean():
    """
    Remove data from example results.
    :return: None
    """
    results = os.path.join(os.getcwd(), 'results')
    if os.path.exists(results):
        shutil.rmtree(results)
    os.mkdir(results)


def clean(method):
    def inner(*args, **kwargs):
        _clean()
        method(*args, **kwargs)
        _clean()
    return inner


def pareto_bool(
        y: np.ndarray,
        strict: bool = False,
        omax: Optional[Iterator[bool]] = None,
) -> np.ndarray:
    """
    Generate the Pareto front mask for an array.
    :param y: The input of shape n_samples x m_dimensionality
    :param strict: will not include points that are in between points on the
    Pareto front.
    :param omax: An iterator of bools determining which objectives should be
    maximized. If None, all will be maximized.
    :return:
    """

    # Flip signs if needed
    y = np.copy(y)
    omax = np.full(y.shape[1], True) if omax is None else omax
    mask = ~np.array([omax, ] * len(y))
    y[mask] = -y[mask]

    # Calculate Pareto bool
    comp = operator.le if strict else operator.lt
    return ~np.array([np.any(np.prod(comp(y[i], np.delete(y, i, axis=0)), axis=1, dtype=bool)) for i in range(len(y))])


def pareto_bool_long(
        y: np.ndarray,
        strict: bool = False,
        omax: Optional[Iterator[bool]] = None,
        bin_size: int = 1000,
):
    """
    Generate the Pareto front mask for an array. This function should be caleed
    instead of pareto_bool if the list of values is very long (i.e. > 10,000).
    :param y: The input of shape n_samples x m_dimensionality
    :param strict: will not include points that are in between points on the
    Pareto front.
    :param omax: An iterator of bools determining which objectives should be
    maximized.
    :param bin_size: Bin size for recuve
    None, all will be maximized.-
    :return: None
    """

    return _pareto_bool_recurve(
        y=y,
        tlen=y.shape[0],
        idx=np.arange(y.shape[0]),
        strict=strict,
        omax=omax,
        rbin=bin_size,
    )


def _pareto_bool_recurve(
        y: np.ndarray,
        tlen: int,
        idx: np.ndarray,
        strict: bool = False,
        omax: Optional[Iterator[bool]] = None,
        rbin: int = 1000,
):
    """
    Generate the Pareto front mask for an array.
    :param y: The input of shape n_samples x m_dimensionality
    :param tlen: The total length of the initial array
    :param idx: The indexes of the points passed
    :param strict: will not include points that are in between points on the
    Pareto front.
    :param omax: An iterator of bools determining which objectives should be
    maximized. If
    :param rbin: Bin size for recuve
    None, all will be maximized.
    :return: pbool
    """

    # Get the length
    ilen = y.shape[0]

    # Number of workers needed
    num_workers = int(ilen / rbin) + (ilen % rbin > 0)

    # Create pool
    pool = mp.Pool()

    # Create results storage
    results = list()

    # For each worker
    for n in range(num_workers):
        # Define job
        i = n * rbin
        j = (n + 1) * rbin
        results.append(pool.apply_async(
            pareto_bool,
            kwds=dict(
                y=y[i:j, :],
                strict=strict,
                omax=omax,
            )
        ))

    # Complete
    for result in tqdm(results):
        result.get()
    pool.close()
    pool.join()

    # Get pbool
    pbool = np.concatenate([r.get() for r in results])

    # Log the state
    current = np.sum(pbool)
    print(
        f'{current}/{tlen} ({round(current / tlen, 3)}%) from {num_workers}'
        f' workers. (in: {len(idx)}. out: {current})')

    # If the recursive is done
    if np.all(pbool):

        # If the final number of Pareto points is larger than the bin_size, they
        # were not compared
        if current >= rbin:

            # Complete one more Pareto bool
            print('Completing total pbool')
            pbool = pareto_bool(y[pbool], strict=strict, omax=omax)
            idx = idx[pbool]
            result = np.repeat(False, tlen)
            result[idx] = True

        # If otherwise confident
        else:

            # Prepare the data processed so far
            result = np.repeat(False, tlen)
            result[idx] = True

        # Return!
        return result

    return _pareto_bool_recurve(
        y=y[pbool],
        tlen=tlen,
        idx=idx[pbool],
        strict=strict,
        omax=omax,
        rbin=rbin
    )


def get_google_drive_authentication():
    """
    Get Google Drive authentication.
    :return: PyDrive Drive access
    """

    # Determine if authentication is complete
    auth = False

    # Define a place to store created credentials
    name = 'credentials.txt'

    # Create gauth instance
    g_login = GoogleAuth()

    # If previously authenticated, load
    if os.path.exists(name):
        g_login.LoadCredentialsFile(name)

        # If exists, confirm not expired
        if g_login.credentials is not None and not g_login.access_token_expired:
            auth = True

    # If not authenticated, prompt authentication flow
    if not auth:
        g_login.LocalWebserverAuth()

    # Save authentication to disk for next time
    g_login.SaveCredentialsFile(name)

    # Create and return Google Drive access
    drive = GoogleDrive(g_login)
    return drive


def upload_file_to_google_drive(
        file_path,
        dest_id: str,
        read_type='r',
):
    """
    Upload a file to a Google Drive location
    :param file_path: path of file to upload
    :param dest_id: Folder ID to upload to.
    :param read_type: How the data should be read. 'r' or 'rb'.
    :return:
    """

    # Authenticate and create drive instance
    drive = get_google_drive_authentication()

    # Open, get name
    with open(file_path, read_type) as f:
        file_name = os.path.basename(f.name)

        # Create drive file
        drive_file = drive.CreateFile(
            metadata=dict(
                title=file_name,
                parents=[dict(id=dest_id)]
            )
        )
        drive_file.SetContentFile(file_path)

        # Upload
        drive_file.Upload()


if __name__ == '__main__':
    pass
