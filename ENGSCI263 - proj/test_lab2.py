# ENGSCI263: Lab Exercise 2
# test_lab2.py
import matplotlib.pyplot as plt

from uuuoooo import *
import numpy as np


def test_ode_model():
    assert ode_model(3, 10, 4, 3, 6, 20) == 12
    assert ode_model(3, 10, 14, 3, 6, 20) == 42


def function1(t, y,q,pars):
    """
    Returns derivation function cos(t)
    """
    return np.cos(t)


def test_solve_ode():
    """
    Test if function solve_ode is working properly by comparing it with a known result.

    Remember to consider any edge cases that may be relevant.
    """
    tol = 1.e-20
    t1 = np.arange(0, 1.2, 0.2)
    q=[0]*len(t1)

    t,x=solve_ode(function1, 0, 1, 0.2, 0, q, [2])

    # Compare with HAND SOLVED solution
    handSolved = np.array([0, 0.198,0.388,0.563,0.715,0.839])

    assert x.round(2).all() == handSolved.all()
    assert t.all()==t1.all()

test_ode_model()
test_solve_ode()