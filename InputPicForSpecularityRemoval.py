"""
A simple demo file explaining how to use the method to remove highlights in images.
    Resuls are stored in results/ directory
"""

import logging
import specularity_removal


# change the log level accordingly
logging.basicConfig(level=logging.DEBUG)

# Example 1
# Images taken in different view points. In the order of alignment
fnames = [

    'data/38.png',
    'data/39.png'


]
specularity_removal.remove_specularity(fnames)

