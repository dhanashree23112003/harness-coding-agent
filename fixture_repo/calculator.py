def divide(x: float, y: float) -> float:
    if y == 0:
        raise ValueError("Cannot divide by zero")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("Both arguments must be numbers")
    return x / y