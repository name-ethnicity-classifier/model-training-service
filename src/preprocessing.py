import json
import os
import random
import re
import string
import pickle
import unicodedata
from dotenv import load_dotenv
from s3 import S3Handler
from schemas import UntrainedModel, ProcessedName
from config import config, Environment


load_dotenv()


letter_vocabular = string.ascii_lowercase + " " + "-"


def load_dataset() -> dict:
    raw_dataset = S3Handler.get(config.base_data_bucket, "raw_dataset.pickle")

    # shorten dataset to speed up during development
    if config.environment == Environment.DEV:
        raw_dataset = {k: v[:1000] for k, v in raw_dataset.items()}

    nationalities = S3Handler.get(config.base_data_bucket, "nationalities.json")

    return raw_dataset, nationalities


def remove_name_prefix(name: str) -> str:
    prefixes_to_disgard = ("dr", "mr", "ms", "miss", "mrs", "prof")

    if name.startswith(prefixes_to_disgard):
        name = " ".join(name.split(" ")[1:])
        name = remove_name_prefix(name)
        
    return name


def normalize_name(name: str) -> str:
    name = name.lower()
    name = name.replace("ä", "ae")
    name = name.replace("ö", "oe")
    name = name.replace("ü", "ue")

    name = u"{}".format(name)
    name = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("utf-8")
    name = re.sub("[^A-Za-z -]+", "", name)

    return remove_name_prefix(name.strip())


def numerize_name(name: str) -> list[int]:
    return [letter_vocabular.index(char) for char in name]


def process_name(class_index: int, name: str) -> list[int]:
    name = normalize_name(name)
    return [class_index, numerize_name(name)]


def balance_classes(selected_classes: dict):
    # Get the amount of names of the nationality with the least amount of names for balancing
    amount = min([len(names) for nationality, names in selected_classes.items() if nationality != "else"])

    for k, v in selected_classes.items():
        random.shuffle(v)
        selected_classes[k] = v[:amount]


def get_selected_nationalities(selected_classes: list[str], dataset: dict) -> dict:
    if "else" not in selected_classes:
        return {nat: names for nat, names in dataset.items() if nat in selected_classes}

    selected_dataset = {"else": []}
    for nationality_name in dataset:
        if nationality_name in selected_classes:
            selected_dataset[nationality_name] = dataset[nationality_name]
        else:
            selected_dataset["else"].extend(dataset[nationality_name])

    return selected_dataset


def get_selected_groups(groupings: list[str], dataset: dict, available_classes: list[str]) -> dict:
    selected_classes = list(groupings.keys())
    selected_dataset = {c: [] for c in selected_classes}

    if "else" in selected_classes:
        selected_dataset |= {"else": []}

    for nationality_name in available_classes:
        for group_name in groupings:
            if nationality_name in groupings[group_name]:
                selected_dataset[group_name].extend(dataset[nationality_name])
            elif "else" in selected_classes:
                selected_dataset["else"].extend(dataset[nationality_name])

    return selected_dataset


def validate_classes(available_classes: list[str], chosen_classes: list[str]):
    if len(chosen_classes) < 2:
        raise ValueError(f"At least two clases must be selected (or onc class and 'else')")

    invalid_classes = [class_ for class_ in chosen_classes if not class_ in available_classes]
    if len(invalid_classes) > 0:
        raise ValueError(f"The selected classes {', '.join(invalid_classes)} do not exist.")
    

def preprocess_nationalities(classes: list[str]) -> list[ProcessedName]:
    raw_dataset, available_classes = load_dataset()
    
    available_nationalities = available_classes["nationalities"] + ["else"]
    validate_classes(available_nationalities, classes)

    selected_nationalities = get_selected_nationalities(classes, raw_dataset)
    balance_classes(selected_nationalities)

    processed_dataset = []
    for nationality_name in selected_nationalities:
        class_index = classes.index(nationality_name)
        names = selected_nationalities[nationality_name]

        processed_names = list(map(lambda name: process_name(class_index, name), names))
        processed_dataset.extend(processed_names)

    random.shuffle(processed_dataset)

    return processed_dataset


def preprocess_groups(classes: list[str]) -> list[ProcessedName]:
    raw_dataset, available_classes = load_dataset()
    
    # add class for 'else'
    available_classes["nationality_groups"]["else"] = []

    available_groups = list(available_classes["nationality_groups"].keys())
    validate_classes(available_groups, classes)

    # Get just the groupings of the selected groups
    groupings = {key: available_classes["nationality_groups"][key] for key in classes}

    # Get all names of each group indexed the the group name and balance
    selected_groups = get_selected_groups(groupings, raw_dataset, available_classes["nationalities"])
    balance_classes(selected_groups)

    processed_dataset = []
    for group_name in selected_groups:
        class_index = classes.index(group_name)
        names = selected_groups[group_name]

        processed_names = list(map(lambda name: process_name(class_index, name), names))
        processed_dataset.extend(processed_names)

    random.shuffle(processed_dataset)

    return processed_dataset



def create_dataset(model: UntrainedModel):
    classes = list(set(model.classes))
    if model.is_grouped:
        dataset = preprocess_groups(classes)
    else:
        dataset = preprocess_nationalities(classes)

    return dataset

