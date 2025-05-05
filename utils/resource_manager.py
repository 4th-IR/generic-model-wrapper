""" A script of utility resource managers """

#external 
import time 
import os
import psutil
from functools import wraps

#internal 
from utils.logger import get_logger


LOG = get_logger(__name__)

def timer(func):
    """ Takes a function and return how long the function runs 
    Args:
        func (object)

    Returns:
        
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        LOG.info(f'{func.__name__} took {end-start} seconds to run')
        print(f'{func.__name__} took {end-start} seconds to run')

    return wrapper


def memory_monitor(func):
    """ Decorator that logs the memory usage of a function. """
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        LOG.info(f'{func.__name__} used {mem_after - mem_before:.2f} MB of memory.')
        print(f'{func.__name__} used {mem_after - mem_before:.2f} MB of memory.')
        return result
    return wrapper


def progress_callback(progress_bar, bytes_uploaded):
    # Update the progress bar: subtract the previously updated value.
    progress_bar.update(bytes_uploaded - progress_bar.n)