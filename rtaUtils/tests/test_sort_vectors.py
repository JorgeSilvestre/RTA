import pandas as pd
import numpy as np
import math
import pytest

import sys
sys.path.append(".")

from ..sort_vectors import calculate_rotation, distance


def test_trivial_right():
    tracks = pd.Series([0,1,2,3,4,5])
    assert calculate_rotation(tracks) == 5
def test_trivial_left():
    tracks = pd.Series([356,354,352,350])
    assert calculate_rotation(tracks) == -6
def test_value_change():
    tracks = pd.Series([350, 354, 356, 358, 0, 4])
    assert calculate_rotation(tracks) == 14
def test_alternate_values():
    tracks = pd.Series([350, 10, 340, 25, 330, 40, 300, 60, 280, 80])
    assert calculate_rotation(tracks) == 90
def test_jump_left():
    tracks = pd.Series([10, 6, 2, 358, 356, 354])
    assert calculate_rotation(tracks) == -16
def test_jump_right():
    tracks = pd.Series([350, 356, 2, 8, 14])
    assert calculate_rotation(tracks) == 24
def test_two_loops_right():
    tracks = pd.Series([350, 45, 90, 180, 270, 350, 25])
    assert calculate_rotation(tracks) == 395


# def test_distance():
#     array = np.array([(x,x) for x in range(10)])
#     print(array)
#     print(array.shape)

#     assert distance(array) == ([0] + [math.sqrt(2)] * 9)