def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)

def test_non_numeric_input():
    with pytest.raises(ValueError):
        divide(10, 'a')