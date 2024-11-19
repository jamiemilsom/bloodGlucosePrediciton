def float_to_time(f: float) -> str:
    """Convert float time to hh:mm format with precision."""
    is_negative = f < 0
    f = abs(f)

    hours = int(f)
    minutes = round((f - hours) * 100)

    if minutes >= 60:
        hours += minutes // 60
        minutes = minutes % 60

    time_str = f"{hours}:{minutes:02d}"
    return f"-{time_str}" if is_negative else time_str

def time_to_float(time_str: str) -> float:
    """Convert hh:mm format to float"""
    if ':' not in time_str:
        raise ValueError('Time must be in hh:mm format')
    if time_str.count(':') != 1:
        raise ValueError('Time must be in hh:mm format')
    hours, minutes = map(int, time_str.split(':'))
    if hours >= 0:
        return round(hours + minutes / 100, 2)
    else:
        return round(hours - minutes / 100, 2)
    
def float_time_minus(time1: float, time2: float) -> float:
    """Subtract two float times
     e.g., 2.55 - 1.55 = 1.00
     e.g., -2.55 - -1.55 = -1.00
     e.g., 2.55 - -1.55 = 4.50
     """

    hours1 = int(time1)
    minutes1 = int(round((time1 - hours1) * 100))

    hours2 = int(time2)
    minutes2 = int(round((time2 - hours2) * 100))


    total_minutes1 = hours1 * 60 + minutes1
    total_minutes2 = hours2 * 60 + minutes2

    diff_minutes = total_minutes1 - total_minutes2

    result_hours, result_minutes = divmod(abs(diff_minutes), 60)

    if diff_minutes < 0:
        result_hours = -result_hours
        result_minutes = -result_minutes

    return result_hours + result_minutes / 100


def float_time_range(start: float, stop: float, step: float) -> list[str]:
    """
    Generate a range of times in hh:mm format from float inputs.
    
    Args:
        start: Start time in decimal format (e.g., 5.55 for 5:55)
        stop: Stop time in decimal format (e.g., 5.30 for 5:30)
        step: Step size in decimal format (can be negative for counting down)
    
    Returns:
        List of times in hh:mm format
    
    Example:
        float_time_range(5.55, -0.01, -0.05) returns
        ['5:55', '5:50', ..., '-0:05', '-0:10']
    """
    if step == 0.00:
        raise ValueError("Step size cannot be zero")
    
    if step > 0 and start > stop:
        raise ValueError("Ascending range requires start <= stop")
    
    elif step < 0 and start < stop:
        raise ValueError("Descending range requires start >= stop")
    
    if abs(step - int(step)) > 0.59:
        raise ValueError("Step size minutes cannot be larger than 59 minutes")

    times = []
    current = start
    
    while (step > 0 and current <= stop) or (step < 0 and current >= stop):
        times.append(float_to_time(current))

        hours = int(current)
        minutes = int(round((current - hours) * 100))

        step_hours = int(step)
        step_minutes = int(round((step - step_hours) * 100))

        total_minutes = hours * 60 + minutes + (step_hours * 60 + step_minutes)
        next_hours, next_minutes = divmod(total_minutes, 60)

        if total_minutes < 0:
            next_hours, next_minutes = divmod(abs(total_minutes), 60)
            next_hours = -next_hours
            next_minutes = -next_minutes if next_minutes != 0 else 0

        current = next_hours + next_minutes / 100

    return times
    

    