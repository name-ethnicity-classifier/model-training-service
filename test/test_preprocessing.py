import os
from unittest.mock import patch
import pytest
import json
from preprocessing import preprocess_nationalities, preprocess_groups, normalize_name, numerize_name, balance_classes



def load_raw_dataset():
    with open(f"./test/mock/raw_dataset.json", "rb") as o:
        return json.load(o)

def load_classes():
    with open(f"./test/mock/classes.json", "r") as f:
        return json.load(f)

def load_mock_dataset():
    return load_raw_dataset(), load_classes()


@pytest.fixture
def mock_loading_dataset():
    with patch("preprocessing.load_dataset", load_mock_dataset):
        yield


@pytest.mark.it("should create nationality dataset")
def test_create_dataset(mock_loading_dataset):
    selected_classes = ["german", "greek", "vietnamese", "indonesian", "columbian"]
    dataset = preprocess_nationalities(selected_classes)

    names_per_class = 5

    assert len(dataset) == len(selected_classes) * names_per_class
    assert isinstance(dataset[0][0], int)      # class index
    assert isinstance(dataset[0][1], list)     # numerized name


@pytest.mark.it("should create nationality-group dataset")
def test_create_grouped_dataset(mock_loading_dataset):
    selected_classes = ["european", "asian", "southAmerican"]
    dataset = preprocess_groups(selected_classes)
    
    names_per_class = 5 * 2

    assert len(dataset) == len(selected_classes) * names_per_class
    assert isinstance(dataset[0][0], int)      # class index
    assert isinstance(dataset[0][1], list)     # numerized name


@pytest.mark.it("should fail to create dataset with one class")
def test_create_grouped_dataset(mock_loading_dataset):
    selected_classes = ["german", "german"]
    dataset = preprocess_nationalities(selected_classes)
    
    assert len(dataset) == 5


@pytest.mark.it("should normalize name to latin remove prefixes")
def test_name_normalization():
    actual_name = normalize_name("Prof Dr Test Tester")
    expected_name = "test tester"
    assert actual_name == expected_name

    actual_name = normalize_name("Äugustin Öhm")
    expected_name = "aeugustin oehm"
    assert actual_name == expected_name

    actual_name = normalize_name("Têst1 123")
    expected_name = "test"
    assert actual_name == expected_name


@pytest.mark.it("should numerize name correctly")
def test_name_numerization():
    actual_numeric_name = numerize_name("martin baumgatner-schmidt")
    epected_numeric_name = [12, 0, 17, 19, 8, 13, 26, 1, 0, 20, 12, 6, 0, 19, 13, 4, 17, 27, 18, 2, 7, 12, 8, 3, 19]

    assert actual_numeric_name == epected_numeric_name


@pytest.mark.it("should balance classes to have same amount of names per class")
def test_class_balancing():
    classes = {
        "class1": ["name1", "name2", "name3", "name4"],
        "class2": ["name1", "name2", "name3"],
        "class3": ["name1", "name2"],
        "else": ["name1"],              # 'else' will be ignored during balancing
    }

    balance_classes(classes)
    amount_per_class = [len(class_) for class_ in classes.values()]

    assert amount_per_class == [2, 2, 2, 1]


