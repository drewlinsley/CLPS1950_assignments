#!/usr/bin/python -tt
"""
Preliminaries for CLPS1950 programming bootcamp.

This script will introduce you to several concepts:


(1) Imports
(2) Functions
(3) Conditionals
(4) Data types
(5) While loops and the special case for loops
(6) Running a deep learning demo


"""

"""
## (0) Python code formatting
## Associated reading: https://medium.com/@hoffa/400-000-github-repositories-1-billion-files-14-terabytes-of-code-spaces-or-tabs-7cfe0b5dd7fd
"""

# Indentation matters in python, and it relays code structure to your python interpreter.
# There are many different styles of indentation: nesting blocks of code with
# 2 spaces, 4 spaces, or tabs. This script (and my preferred style) uses 4 spaces.
# Use whatever style you'd like, but stay consistent!


"""
## (1) Library imports
## Associated reading: https://developers.google.com/edu/python/introduction
"""
print '\n\n%s\n--------------------\n' % '(1) Library imports'
import os  # One import per line
import numpy as np  # The "as" keyword is an alias for the imported library. See below.
from keras_demo import deep_dream  # Here we import the function deep_dream from the module keras_demo


"""
## (2) Create functions
## Associated reading: https://developers.google.com/edu/python/introduction
"""
print '\n\n%s\n--------------------\n' % '(2) Create functions'


def main(name):
    """This is a Docstring for the function main, which prints hello world.

    Docstrings are easy ways to provide documentation for your code.
    They are encapsulated within two sets of three quotations.

    All of your docstrings should contain (1) A one-line description
    of your code. (2) A description of the inputs. (3) a description
    of the outputs.

    See the following link for a style guide:
    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

    Parameters
    ----------
    name: str

    Returns
    ----------
    success: boolean
    """
    try:  # Try the following code but do not exit upon an error
        print 'Domo arigato, ' + name  # Concatenate strings
        print 'Thank you very much. %s' % name  # Format a string with name inserted into %s
        success = True
    except:
        success = False
    return success


def print_section(section):
    """Print a message that labels a portion of code.

    Parameters
    ----------
    name: section

    Returns
    ----------
    (none): (none)
    """
    print '\n\n%s\n--------------------\n' % section


# Run your function
success = main('Mr. Roboto')

"""
## (3) Conditionals
## Associated reading: https://developers.google.com/edu/python/introduction
"""
print_section('(3) Conditionals')
if success:  # If success is true:
    print 'I want to thank you, please, thank you!'
else:  # Otherwise:
    print 'Kilroy!'


"""
## (4) Data types, associated readings:
## a. https://developers.google.com/edu/python/strings
## b. https://developers.google.com/edu/python/lists
## c. https://developers.google.com/edu/python/dict-files
"""
print_section('(4) Data types')
# Strings
i_am_a_string = 'I am the modern man'
and_can_be_concatenated = i_am_a_string + ' (secret secret I\'ve got a secret).'
and_can_be_concatenated += '\nWho hides behind a mask.'
try:
    but_do_not_combine_me_with_integers += 2
except Exception as e:
    print 'Adding a string with an integer raises the following error: %s' % e
print and_can_be_concatenated

# Lists
lists_are_ordered_structures = [
    'Lists are great',
    'for storing strings that you want',
    'in a specific order']
print ' '.join(lists_are_ordered_structures)  # This combines the list and separates each string with a space
lists_are_ordered_structures = [
    1,
    3,
    5
]
print 'Forward pass through the list: %s' % lists_are_ordered_structures
print 'Backwards pass through the list: %s' % lists_are_ordered_structures[::-1]  # This means reverse
print 'Indexing the second element: %s' % lists_are_ordered_structures[1] # Indexing begins at 0
print 'Another approach to indexing the second element: %s' % lists_are_ordered_structures[-2] # Second from the end

# Dicts
dicts_are_unordered_lookup_tables = {
    'field_1': 2,  # this syntax attaches a value (2) to a key ('field_1')
    'field_2': 'stuff',
    3: 4  # You can mix types
}

# Bonus: Tuples, which are immutable (fixed size) versus the dynamic lists from above. This makes them faster than lists.
my_tuple = ('f1', 'f2')

# Floating points versus integers
output = 2 + 3.  # Use a decimal to indicate float versus int in Python syntax
np_output = np.asarray(2 + 3., dtype=np.float32)  # Or use numpy to explicitly cast


"""
## (5) While loops and the special case of for loops
## Associated reading: https://developers.google.com/edu/python/dict-files
"""
print_section('(5) While loops and the special case of for loops')
count = 0  # Int
limit = 10  # Int
while count < limit:  # While loop
    main('Mr. Roboto')
    count += 1  # This syntax adds 1 to count

for idx in range(10):  # For loop -- idx loops through the numbers 0-9 (range(10))
    main('Mr. Roboto')

[main('Mr. Roboto') for idx in range(10)]  # List comprehension -- a pythonic form of the above

# Python also offers "Generators", which are objects that will yield a different
# value each time they are accessed. Dictionaries have a built-in generator called
# "iteritems", which will pull out a new key:value pair on each iteration.
print ['%s: %s' % (k, v) for k, v in dicts_are_unordered_lookup_tables.iteritems()]
dictionary_comprehension = {k: v for k, v in dicts_are_unordered_lookup_tables.iteritems()}


"""
## (6) Deep learning demo! It will take a while to download the data.
"""
print_section('(6) Deep learning demo!')
deep_dream.run_demo(
    os.path.join(  # This produces a full file pointer to pretty_dog.jpg
        'images',
        'small_pretty_dog.jpg'),  # Feel free to change this image
    output_file='dream_dog')
