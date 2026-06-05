def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError):
        divide(1, 0)

def test_non_numeric_input_raises_value_error():
    with pytest.raises(ValueError):
        divide("a", 1)