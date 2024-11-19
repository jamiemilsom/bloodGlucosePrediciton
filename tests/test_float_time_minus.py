import pytest
from src.helper.helper import float_to_time, time_to_float, float_time_minus

def test_positive_value():
    result = float_time_minus(2.55, 1.55)
    expected = 1.00
    assert result == expected
    
def test_negative_value():
    result = float_time_minus(-2.55, -1.55)
    expected = -1.00
    assert result == expected
    
def test_positive_negative():
    result = float_time_minus(2.55, -1.55)
    expected = 4.50
    assert result == expected

def test_negative_positive():
    result = float_time_minus(-2.55, 1.55)
    expected = -4.50
    assert result == expected
    