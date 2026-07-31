import os
import pickle

def save(state_dict, path: str):
  """Saves the model's parameters to a file."""
  dirname = os.path.dirname(path)
  if dirname: os.makedirs(dirname, exist_ok=True)
  with open(path, 'wb') as f:
    pickle.dump(state_dict, f)

def load(path: str):
  """Loads the model's parameters from a file."""
  with open(path, 'rb') as f:
    state_dict = pickle.load(f)
  return state_dict
