import pytest
from src.helper.helper import time_to_float, float_to_time, float_time_range

def test_descending_range():
    """Test basic descending time range"""
    result = float_time_range(5.55, 5.30, -0.05)
    expected = ['5:55', '5:50', '5:45', '5:40', '5:35', '5:30']
    assert result == expected

def test_ascending_range():
    """Test basic ascending time range"""
    result = float_time_range(5.30, 5.55, 0.05)
    expected = ['5:30', '5:35', '5:40', '5:45', '5:50', '5:55']
    assert result == expected

def test_infinite_range():
    """Test when increment is opposite direction of range"""
    with pytest.raises(ValueError):
        float_time_range(5.55, 6.00, -0.05)

def test_hour_rollover():
    """Test when minutes exceed 60"""
    result = float_time_range(1.55, 2.05, 0.10)
    assert result == ['1:55', '2:05']

def test_negative_times():
    """Test handling of negative time inputs"""
    result = float_time_range(0.15, -0.30, -0.15)
    assert result == ['0:15', '0:00', '-0:15', '-0:30']

def test_invalid_step():
    """Test with zero step size"""
    with pytest.raises(ValueError):
        float_time_range(5.55, 5.30, 0)
        
def test_large_step():
    result = float_time_range(12.59,18.59,2.59)
    assert result == ['12:59','15:58','18:57']
    
if __name__ == "__main__":
    result = float_time_range(12.59,18.59,2.59)
    # Should return '1:55','2:05'
    print(result)