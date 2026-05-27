import os
import pickle
import hashlib
import functools
from utils.data import get_data_dir, _get_dir_signature

CACHE_DIR = os.path.join(get_data_dir(), ".cache")
CACHE_VERSION = "v8"

def disk_cache(func):
    """
    Decorator to cache the results of a function to disk and in memory.
    The cache is automatically invalidated when files in get_data_dir() are modified/added/deleted.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not os.path.exists(CACHE_DIR):
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
            except Exception:
                pass
            
        data_dir = get_data_dir()
        current_sig = _get_dir_signature(data_dir)
        
        # Build a cache key from args and kwargs
        repr_parts = [CACHE_VERSION, func.__module__, func.__name__]
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                repr_parts.append(str(arg))
            elif hasattr(arg, 'shape'):  # pandas DataFrame or numpy array
                # Hash the shape and column names/types to avoid serializing whole dataframe in key
                df_repr = f"df_{arg.shape}"
                if hasattr(arg, 'columns'):
                    df_repr += f"_{list(arg.columns)}"
                repr_parts.append(df_repr)
            else:
                repr_parts.append(f"obj_{id(arg)}")
                
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool, type(None))):
                repr_parts.append(f"{k}:{v}")
            elif hasattr(v, 'shape'):
                df_repr = f"{k}:df_{v.shape}"
                if hasattr(v, 'columns'):
                    df_repr += f"_{list(v.columns)}"
                repr_parts.append(df_repr)
            else:
                repr_parts.append(f"{k}:obj_{id(v)}")
                
        key_str = "_".join(repr_parts)
        key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        
        sig_str = f"{current_sig[0]}_{int(current_sig[1])}"
        cache_filename = f"{func.__name__}_{key_hash}_{sig_str}.pkl"
        cache_path = os.path.join(CACHE_DIR, cache_filename)
        
        # Memory cache key (including current_sig for instant invalidation)
        mem_key = (key_hash, current_sig)
        if not hasattr(func, "_mem_cache"):
            func._mem_cache = {}
            
        if mem_key in func._mem_cache:
            return func._mem_cache[mem_key]
            
        # Try loading from disk
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                func._mem_cache[mem_key] = result
                return result
            except Exception:
                pass
                
        # Cache miss — execute the function
        result = func(*args, **kwargs)
        
        # Try cleaning up old cache files for this function to save space
        try:
            for file in os.listdir(CACHE_DIR):
                if file.startswith(f"{func.__name__}_") and file.endswith(".pkl") and file != cache_filename:
                    os.remove(os.path.join(CACHE_DIR, file))
        except Exception:
            pass
            
        # Save to disk and memory
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            func._mem_cache[mem_key] = result
        except Exception:
            pass
            
        return result
    return wrapper
