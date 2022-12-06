import yaml

class params:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

def get_params(filename):
    with open(filename) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return params(**data)