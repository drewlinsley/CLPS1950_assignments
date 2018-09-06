"""
## Module 1 assignment 1: Python programming


## Fill in your name here: ______


## Grading

You will be graded on two items that you must turn into canvas:
    (1) A completed version of this assignment. [3 points]
    (2) A PDF with answers to questions in parts 0 and 2. [2 points]
Your completed assignment and PDF response to questions must be combined
into a zip and submitted to canvas for credit. See
[example_submission]Drew_Linsley_module_1_assignment_2.zip for an
example of how you should zip up your assignment and PDF writeup, as well
as an example formatting of the PDF writeup.

Your submission for this assignment should be called:

{your_name}_module_1_assignment_1.zip

## Description

This assignment will instruct you on using the Python Numpy library for
mathematical operations. You will first visit the linked tutorials and prepare
short responses to the questions that are asked.

Afterwards, you will complete the four coding exercises below. These
are denoted by EDIT HERE in the associated docstrings.

Estimated time: 45-60 mins.

## Tips

If you are using Anaconda and Sublime, turn on the python linter.
    This will notify you of errors in your code.
    a. Click Sublime text -> Preferences -> Package settings -> Anaconda ->
        Settings - Default
    b. On line 310 set: "anaconda_linter_phantoms": true,
    c. Restart sublime 3.
If you are using a linter, your text editor will provide feedback on:
    a. Syntax errors
    b. Style.
    (a) feedback is critical to fix, but feel free to ignore (b) if it doesn't
    bother you. That said, it is important to maintain a consistent style as
    you code. This will reduce errors and make it easier for others to read.
If you have a GPU and CUDA installed:
    Make sure you "mask" your GPUs before running. Tensorflow (the backend of
    Keras) will automatically spread to all available GPUs, reserving as much
    GPU memory as possible. This means you must mask out GPUs that you don't
    want Tensorflow to see and spread to when running analyses with GPU.

    Do this by running python or ipython in the following way:

    CUDA_VISIBLE_DEVICES=0 python my_script.py

    or

    CUDA_VISIBLE_DEVICES=0 ipython

    where the number 0 refers to the only GPU you want visible for Tensorflow.
    Use the command nvidia-smi to see the available GPUs and their IDs, then
    exchange the 0 above for whatever GPU ID you'd like. You can make multiple
    GPUs visible by replacing the 0 with 0,1 or make no GPUs visible by setting
    a large positive or negative number (e.g. 999).

"""


"""
1. Read the following tutorials and complete the written exercises inside.
    Tutorial 1:
        https://docs.google.com/document/d/1AL8ABHBADUIMzFknYRc1VhGiY7bNQRAvIIJ3gHwe8GU/edit
    Tutorial 2:
        https://docs.google.com/document/d/1cqgvi_6CKUtwFAGIIeFnmweSy_wZ6XKYHaJGoWhalMk/edit

"""

import numpy as np
from matplotlib import pyplot as plt

"""
###########################
2. EDIT HERE!
###########################

Your tasks are the following:

a. Define a 5x5 array of uniform random integers between 1 and 10.
b. Calculate the row-wise and column-wise min, max, sum, mean, and standard
deviation of the array with numpy.
c. Make three separate boolean arrays (i.e. consisting of only
True/False values) that flag properties about the array you generated in
(a). The first identifies values in (a) that are greater than 5;
the second identifies values that are equal to 2; the third identifies
values that are divisible by 3.
d. Index the array you generated in (a) with each of your masks. What values
are returned? This is called logical indexing. Compare its elegance to
creating a for loop that checks the contents of each element in your array.
"""

my_array = np.random.  # part a

my_min = np. (my_array)  # part b
my_max = np. (my_array)  # part b
my_sum = np. (my_array)  # part b
my_mean = np. (my_array)  # part b
my_sd = np. (my_array)  # part b

mask_i = my_array  # part c
mask_ii = my_array  # part c
mask_iii = my_array  # part c

result_i = my_array[]  # part d
result_ii = my_array[]  # part d
result_iii = my_array[]  # part d


"""
###########################
3. EDIT HERE!
###########################

Let N be a positive natural number. Imagine the following recursive
procedure: if N is even, set N = N/2. If N is odd, set N = 3N + 1.
Continue this procedure until N equals 1. The number of steps required
to reach one is called the stopping time. Do this for every integer up to
100000.

For each integer, keep track stopping times in a list. Plot the number of
steps for each integer and give your plot a title and axis labels. Also, make
a histogram with 100 bins of stopping times with a title and labels. Include
this figure in your PDF writeup.

Note: it is currently unknown whether every integer reaches 1 under this
procedure. The hypothesis that every natural number has finite stopping time
is called the Collatz Conjecture.
"""

max_n = 100000
stopping_times = []
for n in range(max_n):
    print ('Testing n = ' + str(n))
    tmp_n = n
    counter = 0
    while tmp_n > 1:

    else:

    stopping_times += [counter]  # Append counter to the stopping_times list

fig = plt.figure()
plt.hist(np.array(stopping_times), bins=100)
plt.title('Collatz Stopping Distribution')
plt.xlabel('Stopping Time')
plt.ylabel('Frequency')
plt.savefig('module_1_assigment_1_Q2_figure.png')


"""
###########################
4. EDIT HERE!
###########################

a. Make an empty list called array_list. For ii between 1 and 10, generate a
10x10 array of uniform (float) random numbers on the interval [0, ii].
Store each array in array_list. Then, cast array_list as an array called
my_array. Print the shape of my_array like "The shape of my_array is SHAPE".

b. Then, create three new arrays.

The first array will have just rows 1, 3, 5, 7, and 9 from dimension 1 of
my_array.

The second array will have just columns 1, 3, 5, 7, and 9 from dimension 2.

The third array will have just layers 1, 3, 5, 7, and 9 from dimension 3.

Display the size of these new arrays with the string "The shape of ARRAY is
SHAPE."
"""
array_list = []  # Part a
for ii in range():  # Part a
    array =  # Part a
    array_list += [array]  # Part a
my_array = np.array(array_list)  # Part a
print('The shape of my_array is (%s, %s, %s)' % np.shape(my_array))  # Part a

dim_range = range()  # Part b
dim0_array = my_array[]  # Part b
dim1_array = my_array[]  # Part b
dim2_array = my_array[]  # Part b
print('The shape of the first array is %s' % np.shape(dim0_array))  # Part b
print('The shape of the second array is %s' % np.shape(dim1_array))  # Part b
print('The shape of the third array is %s' % np.shape(dim2_array))  # Part b


"""
###########################
5. EDIT HERE!
###########################

A magic square is an nxn array of natural numbers such that, the column-sum,
row-sum and diagonal-sum (trace) are all equal.

a. Write a function that generates a magic square using the following
procedure:
    i. Initialize an NxN array of zeros where N is odd. Let n = 1,
    let i = 0 and let j = np.ceil(N / 2.) (the next integer after N / 2).

    ii. Put n in index i, j.

    iii. Increment both i and j by i. If this goes outside the array, wrap
    around to the bottom row and the leftmost column.

    iv. If this new index already has something in it, undo the incrementing
    of i and j and just set i = i + 1.

    v. Go back to step ii. Repeat until n = N^2.

b. Generate a 5x5 magic square. Write a function that checks this is indeed a
magic square.

c. Add your magic square to itself 100 times in sequence, each time checking
if the sum is a magic square.

d. Generate 10 uniform random integers from 0 to 100 and use them to scale
your 5x5 magic square. Is the scaled magic square a magic square?
"""


def make_magic_square(N):  # part a
    """Create a magic square.

    Parameters
    ----------
    N : int
        Shape of the magic square

    Returns
    -------
    numpy int array
        A magic square of N x N
    """
    if N % 2 == 0:
        print('N must be odd.')
    my_magic_square = np.zeros((N, N))
    i = 0
    j = np.ceil(N / 2.).astype(int)
    n = 1
    while n <= N**2:
        my_magic_square[i, j] = n
        n += 1
        i_next =
        j_next =
        if my_magic_square[i_next, j_next] > 0:
            i =
        else:
            i =
            j =
    return my_magic_square


N = 5
my_magic_square = make_magic_square(N)
magic_value = np.sum(my_magic_square, axis=0)[0]


def magic_check(magic_square, magic_value):
    """Check the magic square.

    Parameters
    ----------
    magic_square : numpy int array
        A magic square of N x N
    magic_value : int
        Value for verifying the magic square

    Returns
    -------
    boolean
        Validation of magic_square
    """
    row_sum = np.sum(magic_square, axis=0)
    col_sum = np.sum(magic_square, axis=1)
    diag_sum = np.sum(np.diag(magic_square))
    if :  # part b
        return True
    else:
        return False


success = magic_check(my_magic_square, magic_value)
if success:
    print('It is a magic square.')
else:
    print('It is NOT a magic square.')

tmp_magic_square = np.copy(my_magic_square)  # Make a copy of your magic square
for i in range(100):
    tmp_magic_square =  # Part c
    magic_value =  # Part c
    success = magic_check(tmp_magic_square, magic_value)
    if success:
        print('Passed the %sth check.' % i)
    else:
        print('Failed the %sth check.' % i)

for i in range(10):
    k = np.random.  # Part d
    scaled_magic_square =  # Part d
    magic_value =  # Part d
    success = magic_check(scaled_magic_square, magic_value)
    if success:
        print('Passed the %sth check.' % i)
    else:
        print('Failed the %sth check.' % i)
