import pytest

import sys
sys.path.append(".")

from ..sort_vectors import generate_windows, generate_changes

@pytest.fixture
def sequence():
    return list(range(22))

def test_simple_no_overlap(sequence):
    assert generate_windows(len(sequence), 10, 0, 0, 0) == [(0,0,10),(1,10,20),(2,20,22)]

def test_simple_overlap(sequence):
    assert generate_windows(len(sequence), 10, 5, 0, 0) == [(0,0,10),(1,5,15),(2,10,20),(3,15,22)]

def test_no_overlap(sequence):
    assert generate_windows(len(sequence), 7, 0, 0, 0) == [(0,0,7),(1,7,14),(2,14,21),(3,21,22)]

def test_overlap(sequence):
    assert generate_windows(len(sequence), 7, 3, 0, 0) == [(0,0,7),(1,4,11),(2,8,15),(3,12,19),(4,16,22)]

def test_simple_overlap_start(sequence):
    assert generate_windows(len(sequence), 10, 5, 2, 0) == [(0,2,12),(1,7,17),(2,12,22)]

def test_simple_overlap_start2(sequence):
    assert generate_windows(len(sequence), 10, 5, 1, 0) == [(0,1,11),(1,6,16),(2,11,21),(3,16,22)]

def test_simple_overlap_end(sequence):
    assert generate_windows(len(sequence), 10, 5, 0, 20) == [(0,0,10),(1,5,15),(2,10,20)]

def test_simple_overlap_end2(sequence):
    assert generate_windows(len(sequence), 10, 5, 0, 21) == [(0,0,10),(1,5,15),(2,10,20),(3,15,21)]

def test_overlap_start_end(sequence):
    assert generate_windows(len(sequence), 7, 3, 2, 20) == [(0,2,9),(1,6,13),(2,10,17),(3,14,20)]


def test_changes_simple():
    assert generate_changes(0,5) == [(1,3)]

def test_changes_simple2():
    assert generate_changes(0,10) == [(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,4),(2,5),(2,6),(2,7),(2,8),
                                       (3,5),(3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,7),(5,8),(6,8)]