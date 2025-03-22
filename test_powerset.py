from main import get_powerset

def test_empty_array():
    result = get_powerset([], 0)
    assert result == [[]]

def test_single_element():
    result = get_powerset([1], 0)
    assert sorted(result) == sorted([[], [1]])

def test_two_elements():
    result = get_powerset([1, 2], 0)
    expected = [[], [1], [2], [1, 2]]
    assert sorted(result) == sorted(expected)

def test_three_elements():
    result = get_powerset([1, 2, 3], 0)
    expected = [
        [], [1], [2], [3],
        [1, 2], [1, 3], [2, 3],
        [1, 2, 3]
    ]
    assert sorted(result) == sorted(expected)

def test_duplicate_elements():
    result = get_powerset([1, 1], 0)
    expected = [[], [1], [1], [1, 1]]
    assert sorted(result) == sorted(expected)

def test_negative_numbers():
    result = get_powerset([-1, 0, 1], 0)
    expected = [
        [], [-1], [0], [1],
        [-1, 0], [-1, 1], [0, 1],
        [-1, 0, 1]
    ]
    assert sorted(result) == sorted(expected) 