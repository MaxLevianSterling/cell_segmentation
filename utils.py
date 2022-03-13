import os
import sys


class HiddenPrints:
    """Hides print output"""

    def __enter__(self):
        """Upon entering"""

        # Save stdout parameters
        self._original_stdout = sys.stdout

        # Replace stdout with null
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Upon leaving"""

        # Close stdout replacement
        sys.stdout.close()

        # Restore original stdout parameters
        sys.stdout = self._original_stdout


def path_gen(elmnts, file=False):
    """Generates path string out of elements
    
    Args:
        elmnts (list of strings): path elements
        file (bool): whether last element is a file
 
    Returns:
        (string) merged path string
    """

    # Initialize path string
    path = ''

    # Append elements
    for elmnt in elmnts:
        path += f'{elmnt}/'
    
    # Remove last forward slash if file
    if file:
        path = path[:-1:]

    return path


def unpad(arr, pad_widths):
    """Unpads a numpy array
    
    Args:
        arr (np.array): input array
        pad_widths (int): pad widths to remove
 
    Returns:
        (np.array) unpadded array
    """

    # Initialize array slices
    slices = []

    # For every array dimension
    for c in pad_widths:

        # Take inverse of the far point
        e = None if c[1] == 0 else -c[1]

        # Create slice object for that dimension
        slices.append(slice(c[0], e))
    
    # Unpad
    return arr[tuple(slices)]