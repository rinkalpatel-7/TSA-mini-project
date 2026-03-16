import pytest
from src.preprocessor import generate_synthetic_data

def test_data_length():
    data = generate_synthetic_data()
    assert len(data) == 120  # Simple check for data integrity