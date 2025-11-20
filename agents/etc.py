import sys
from pathlib import Path

def set_path():
    sys.path.append(str(Path(__file__).parents[1]))
    