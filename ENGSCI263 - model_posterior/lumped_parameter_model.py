#########################################################################################
#
# Function library for plotting lumped parameter model
#
# 	Functions:
#		lmp_2D: lumped parameter model with 2 variables
#		lmp_3D: lumped parameter model with 3 variables
#		obj: objective function for a lmp
#
#########################################################################################

# import modules and functions
import numpy as np
from Data_Vis import *

# global variables - observations

tq, q = np.genfromtxt('gs_mass.txt', delimiter=',', skip_header=1).T
tp, p = np.genfromtxt('gs_pres.txt', delimiter=',', skip_header=1).T


# define derivative function
def ode_model(t, x, q, a, b, x0):
    ''' Return the derivative dx/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        x : float
            Dependent variable.
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        x0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> ode_model(0, 1, 2, 3, 4, 5)
        22

    '''
    if (x - x0) > 0.1e+6:
        return (a*q)-(b*((x-x0)**2))
    else:
        return (a*q)

def solve_lpm(f, t, x0, pars):
    ''' Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE should be solved using the Improved Euler Method.

        Function q(t) should be hard coded within this method. Create duplicates of
        solve_ode for models with different q(t).

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    '''
    # setup initial value arrays and function q(t)
    intt = t
    intx = np.zeros(len(intt))
    intx[0] = x0
    dt = intt[1]-intt[0]
    qt = interpolate_massflow(intt)

    for i in range(len(intt) - 1):
        # compute k1
        k1 = f(intt[i], intx[i], qt[i], *pars, x0)
        # compute euler value
        xe = intx[i] + dt * k1
        # compute k2
        k2 = f(intt[i + 1], xe, qt[i], *pars, x0)
        # finally, compute improved euler value
        intx[i + 1] = intx[i] + ((dt / 2) * (k1 + k2))

    return intt, intx/(10**6)


def fit_mvn(parspace, dist):
    """Finds the parameters of a multivariate normal distribution that best fits the data

    Parameters:
    -----------
        parspace : array-like
            list of meshgrid arrays spanning parameter space
        dist : array-like 
            PDF over parameter space
    Returns:
    --------
        mean : array-like
            distribution mean
        cov : array-like
            covariance matrix		
    """

    # dimensionality of parameter space
    N = len(parspace)

    # flatten arrays
    parspace = [p.flatten() for p in parspace]
    dist = dist.flatten()

    # compute means
    mean = [np.sum(dist * par) / np.sum(dist) for par in parspace]

    # compute covariance matrix
    # empty matrix
    cov = np.zeros((N, N))
    # loop over rows
    for i in range(0, N):
        # loop over upper triangle
        for j in range(i, N):
            # compute covariance
            cov[i, j] = np.sum(dist * (parspace[i] - mean[i]) * (parspace[j] - mean[j])) / np.sum(dist)
            # assign to lower triangle
            if i != j: cov[j, i] = cov[i, j]

    return np.array(mean), np.array(cov)
