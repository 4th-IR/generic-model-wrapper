""" A script of utility resource managers """

#external 
import time 
from functools import wraps

#internal 
from utils.logger import get_logger


LOG = get_logger('time')

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


def progress_callback(progress_bar, bytes_uploaded):
    # Update the progress bar: subtract the previously updated value.
    progress_bar.update(bytes_uploaded - progress_bar.n)