"""
## Module 2 assignment 1: Perceptrons and the XOR problem


## Fill in your name here: ______


## Grading

You will be graded on one item that you must turn into canvas:
    (1) A PDF with answers to questions in part 1. [5 points]
Your completed assignment and PDF response to questions must be combined
into a zip and submitted to canvas for credit. See
[example_submission]Drew_Linsley_module_2_assignment_2.zip for an
example of how you should zip up your assignment and PDF writeup, as well
as an example formatting of the PDF writeup.

Your submission for this assignment should be called:

{your_name}_module_2_assignment_1.zip

## Description

Perceptron neural networks are "shallow" one-layer networks that are effective
on certain artificial toy problems. They are not effective, however, on the
kinds of visual tasks that biological visual systems rapidly and effortlessly
carry out on a daily basis in support of everyday behavior: recognizing
complex visual scenes and identifying their constituent objects.

In this assignment you will analyze a key limitation of the perceptron, and
build a solution.

Fill in missing code to complete the assignment. These portions of the code
are denoted with numbered blocks that say "EDIT HERE!".

Estimated time: 30 minutes.

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
1. Complete the following exercises on the limitations of perceptrons.
    https://docs.google.com/document/d/1rrAh_35_jVvnwsTJV2HQ3PwVv9CDCdIOdq4FHClEIy0/edit?usp=sharing
"""
