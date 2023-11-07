import time
import logging as log

def get_random_seed():
    """
    It takes the current time in milliseconds, and then does some bitwise operations to get a random
    seed

    Returns:
      The seed is being returned.
    """
    t = int(time.time() * 1000)
    seed = (
        ((t & 0xFF000000) >> 24)
        + ((t & 0x00FF0000) >> 8)
        + ((t & 0x0000FF00) << 8)
        + ((t & 0x000000FF) << 24)
    )


    return seed
