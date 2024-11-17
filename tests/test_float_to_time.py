import pytest
from src.helper.helper import float_to_time


def test_positive_value():
    result = float_to_time(2.55)
    expected = '2:55'
    assert result == expected

def test_negative_value():
    result = float_to_time(-2.55)
    expected = '-2:55'
    assert result == expected
    
def test_every_minute():
    for minute in range(60): 
        time_float = 0.00 + minute / 100 
        result = float_to_time(time_float)
        expected = f"0:{minute:02}"
        assert result == expected, f"Failed at {time_float}: got {result}, expected {expected}"
        
def test_every_negative_minute():
    for minute in range(1,60): 
        time_float = 0.00 - minute / 100  
        result = float_to_time(time_float)
        expected = f"-0:{minute:02}"
        assert result == expected, f"Failed at {time_float}: got {result}, expected {expected}"

    
def test_invalid_minutes_positive():
    result = float_to_time(5.99)
    expected = '6:39'
    assert result == expected
    
def test_invalid_minutes_negative():
    result = float_to_time(-5.99)
    expected = '-6:39'
    assert result == expected
    
def test_negative_minutes():
    result = float_to_time(-0.23)
    expected = '-0:23'
    assert result == expected
    
def test_high_precision():
    result = float_to_time(0.2456)
    expected = '0:25'
    assert result == expected
    
def test_large_number():
    result = float_to_time(-100000000.24)
    expected = '-100000000:24'
    assert result == expected
    
if __name__ == "__main__":
    result = float_to_time(5.57)
    # Should return '1:55','2:05'
    print(result)