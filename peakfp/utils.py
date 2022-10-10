"""Code snippets utils
"""
import pickle


def load_pickle(path: str):
    """Load pickle from path.
    Args:
        path (str): Pickle path
    Returns:
        Loaded pickle"""
    with open(path, 'rb') as fin:
        return pickle.load(fin)
