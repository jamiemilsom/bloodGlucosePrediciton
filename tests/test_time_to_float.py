import pytest
from src.helper.helper import time_to_float

def test_positive_number():
    result = time_to_float('5:45')
    expected = 5.45
    assert result == expected

def test_negative_number():
    result = time_to_float('-5:45')
    expected = -5.45
    assert result == expected
        
def test_large_number():
    result = time_to_float('100000:45')
    expected = 100000.45
    assert result == expected

def test_wrong_format():
    with pytest.raises(ValueError):
        time_to_float('5.45')