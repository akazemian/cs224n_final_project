import os
import pickle
import yaml
import functools
import gc 

with open("paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

CACHE = paths["CACHE"]
    
def cache(file_name_func):
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(os.path.join(CACHE, 'embeddings'), exist_ok=True)
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            file_name = file_name_func(*args, **kwargs)
            cache_path = os.path.join(CACHE, file_name)

            if os.path.exists(cache_path):
                print(f'activations for {cache_path} already exist')
                return

            result = func(self, *args, **kwargs)
            with open(cache_path, 'wb') as file:
                pickle.dump(result, file)
            gc.collect()

        return wrapper
    return decorator
