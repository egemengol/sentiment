import os
from pathlib import Path


DATA_FOLDER = Path("./data").absolute()


def docs(dataset="train", feel: str = "neg"):
    # Document generator
    for filename in os.scandir(DATA_FOLDER / dataset / feel):
        if filename.name.startswith("."):
            continue
        yield DATA_FOLDER / dataset / feel / filename
