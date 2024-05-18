from pickle import dump, load


def load_model(name: str) -> any:
    """Load a model from a file"""

    with open(f"./models/{name}", 'rb') as file:
        return load(file)


def save_model(model: any, name: str) -> None:
    """Save the model to a file"""

    with open(f"./models/{name}", 'wb') as file:
        dump(model, file)
