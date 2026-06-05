"""Happy-path tests for calculator.py.

These pass before validation is added. The agent must extend this file
with tests for ValueError (divide by zero, non-numeric input).
"""
import pytest
from calculator import add, divide, multiply, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(10, 4) == 6


def test_multiply():
    assert multiply(3, 7) == 21


def test_divide_basic():
    assert divide(10, 2) == 5.0


def test_divide_float():
    assert divide(1, 4) == 0.25
