from pathlib import Path

def _get_path_if_exists(path, filename):
    if (path / filename).exists():
        return path / filename
    return False

def find_path(filename):
    ## => useful code if ran in the main script
    #if "__file__" in globals():
    #    path = Path(__file__).parent
    #else:
    #    path = Path.cwd()
    ## => but here __file__ is always utils.py
    path = Path(__file__).parent
    
    data = Path("data")
    p = False
    while p == False:
        p = _get_path_if_exists(path, filename) or \
            _get_path_if_exists(path, data / filename)
        if p == False and path.parent == path:
            raise FileNotFoundError(f"{filename} not found")
        path = path.parent
    
    return p
