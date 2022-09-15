# ENGSCI263: Lumped Parameter Models
# main.py

# PURPOSE:
# To COMPUTE a posterior distribution and use it to CONSTRUCT an ensemble of models.
# To EXAMINE structural error.

# PREPARATION:
# Notebook uncertainty.ipynb

# SUBMISSION:
# There is NOTHING to submit for this lab.

# INSTRUCTIONS:
# Jump to section "if __name__ == "__main__":" at bottom of this file.

# import modules and functions
import matplotlib.pyplot as plt
import numpy as np
from plotting import *
from Data_Vis import *
from lumped_parameter_model import *
from uuuoooo import cumulator

####################################################################################
#
# Task 1: Ad hoc calibration of the lumped parameter model.
#
####################################################################################
def get_familiar_with_model():
    ''' This function runs and plots the lumped parameter model for your selection of
        model parameters.
    '''
    # **to do**
    # 1. CHOOSE values of a and b that provide a good fit to the model
    # *stuck?* look at Section 3.4.2 in the Uncertainty notebook
    # 2. CALCULATE the sum of squares objective function
    # 3. ANSWER the questions in the lab document

    # set model parameters (we'll leave c=0 for now)
    # a=
    # b=
    a = 7.5
    b = 8e-6

    # get data and run model
    # po = pressure observation
    tp, po = load_pressure_data()
    po = po / (10 ** 6)
    # pm = pressure model
    tm, pm = solve_ode(ode_model, 2009, 2019, 0.01, 25.16e+6, [a, b])

    # error variance - 2 bar
    v = 2.

    # 2. calculate the sum-of-squares objective function
    # S =
    newpm = np.array([])
    for i in range(len(tm)):
        if (np.round(tm[i], 2) in tp):
            newpm = np.append(newpm, pm[i])
    pm = newpm

    S = np.sum((po - pm) ** 2) / v

    # plotting commands
    f, ax = plt.subplots(1, 1)
    ax.plot(tp, pm, 'b-', label='model')
    ax.errorbar(tp, po, yerr=v, fmt='ro', label='data')
    ax.set_xlabel('time [year]')
    ax.set_ylabel('pressure [MPa]')
    ax.set_title('objective function: S={:3.2f}'.format(S))
    ax.legend()
    plt.show()


####################################################################################
#
# Task 2: Grid search to construct posterior.
#
####################################################################################
def grid_search():
    ''' This function implements a grid search to compute the posterior over a and b.

        Returns:
        --------
        a : array-like
            Vector of 'a' parameter values.
        b : array-like
            Vector of 'b' parameter values.
        P : array-like
            Posterior probability distribution.
    '''
    # **to do**
    # 1. DEFINE parameter ranges for the grid search
    # 2. COMPUTE the sum-of-squares objective function for each parameter combination
    # 3. COMPUTE the posterior probability distribution
    # 4. ANSWER the questions in the lab document

    # 1. define parameter ranges for the grid search
    a_best = 7.5
    b_best = 8e-6

    # number of values considered for each parameter within a given interval
    N = 51

    # vectors of parameter values
    a = np.linspace(a_best / 2, a_best * 1.5, N)
    b = np.linspace(b_best / 2, b_best * 1.5, N)

    # grid of parameter values: returns every possible combination of parameters in a and b
    A, B = np.meshgrid(a, b, indexing='ij')

    # empty 2D matrix for objective function
    S = np.zeros(A.shape)

    # data for calibration
    tp, po = load_pressure_data()
    po = po / (10 ** 6)

    # error variance - 2 bar
    v = 2.

    # grid search algorithm
    for i in range(len(a)):
        for j in range(len(b)):
            # 3. compute the sum of squares objective function at each value
            # pm =
            # S[i,j] =
            tm, pm = solve_ode(ode_model, 2009, 2019, 0.01, 25.16e+6, [a[i], b[j]])
            newpm = np.array([])
            for k in range(len(tm)):
                if (np.round(tm[k], 2) in tp):
                    newpm = np.append(newpm, pm[k])
            pm = newpm
            S[i, j] = np.sum((po - pm) ** 2) / v

    # 4. compute the posterior
    # P=
    P = np.exp(-S / 2.)

    # normalize to a probability density function
    Pint = np.sum(P) * (a[1] - a[0]) * (b[1] - b[0])
    P = P / Pint

    # plot posterior parameter distribution
    plot_posterior(a, b, P=P)

    return a, b, P


####################################################################################
#
# Task3: Open fun_with_multivariate_normals.py and complete the exercises.
#
####################################################################################

####################################################################################
#
# Task 4: Sample from the posterior.
#
####################################################################################
def construct_samples(a, b, P, N_samples):
    ''' This function constructs samples from a multivariate normal distribution
        fitted to the data.

        Parameters:
        -----------
        a : array-like
            Vector of 'a' parameter values.
        b : array-like
            Vector of 'b' parameter values.
        P : array-like
            Posterior probability distribution.
        N_samples : int
            Number of samples to take.

        Returns:
        --------
        samples : array-like
            parameter samples from the multivariate normal
    '''
    # **to do**
    # 1. FIGURE OUT how to use the multivariate normal functionality in numpy
    #    to generate parameter samples
    # 2. ANSWER the questions in the lab document

    # compute properties (fitting) of multivariate normal distribution
    # mean = a vector of parameter means
    # covariance = a matrix of parameter variances and correlations
    A, B = np.meshgrid(a, b, indexing='ij')
    mean, covariance = fit_mvn([A, B], P)

    # 1. create samples using numpy function multivariate_normal (Google it)
    # samples=
    samples = np.random.multivariate_normal(mean, covariance, N_samples)
    # plot samples and predictions
    plot_samples(a, b, P=P, samples=samples)

    f,ax = plt.subplots(2,1)
    samples1 = np.random.multivariate_normal(mean, covariance, 10000)
    ax[0].hist(samples1[:, 0])
    ax[0].set_title('Histogram of lumped parameter a')
    ax[1].hist(samples1[:, 1])
    ax[1].set_title('Histogram of lumped parameter b')
    plt.tight_layout()
    plt.show()

    return samples


####################################################################################
#
# Task 5: Make predictions for your samples.
#
####################################################################################
def model_ensemble(samples):
    ''' Runs the model for given parameter samples and plots the results.

        Parameters:
        -----------
        samples : array-like
            parameter samples from the multivariate normal
    '''
    # **to do**
    # Run your parameter samples through the model and plot the predictions.

    # 1. choose a time vector to evaluate your model between 1953 and 2012
    # t =
    t0, t1, dt, x0 = 2009, 2019, 1 / 12, 25.16e+6
    t = np.arange(t0, t1+dt, dt)

    # 2. create a figure and axes (see TASK 1)
    # f,ax =
    f, ax = plt.subplots(1, 1)
    f5, ax5 = plt.subplots(1, 1)
    ax5.set_title('Uncerainty prediction of gas leakage')
    ax5.set_ylabel('Pressure(MPa)')
    ax5.set_xlabel('Time(years)')
    # 3. for each sample, solve and plot the model  (see TASK 1)
    tp, po = np.genfromtxt('gs_pres.txt', delimiter=',', skip_header=1).T
    mtime1, mpres1 = solve_ode(ode_model, t0, t1, dt, x0, [7.5, 8e-6])
    cumtime, cumgl = cumulator(mpres1*10**6,x0,0.1e+6,mtime1,mtime1[0],mtime1[-1])

    timex = 2023
    for a, b in samples:
        # pm=
        # ax.plot(
        # *hint* use lw= and alpha= to set linewidth and transparency

        mtime, mpres = solve_ode(ode_model, t0, t1, dt, x0, [a, b])
        t2, p2 = predict_t(ode_model, t0, t1, timex, dt, x0, mpres[-1] * (10 ** 6), [a, b], 2)
        cumtp1, cumgl1 = cumulator(np.append(mpres, p2) * (10 ** 6), x0, 0.1e+6, np.append(mtime, t2), t1, timex)

        t3, p3 = predict_t(ode_model, t0, t1, timex, dt, x0, mpres[-1] * (10 ** 6), [a, b], 1.5)
        cumtp2, cumgl2 = cumulator(np.append(mpres, p3) * (10 ** 6), x0, 0.1e+6, np.append(mtime, t3), t1, timex)

        t4, p4 = predict_t(ode_model, t0, t1, timex, dt, x0, mpres[-1] * (10 ** 6), [a, b], 1)
        cumtp3, cumgl3 = cumulator(np.append(mpres, p4) * (10 ** 6), x0, 0.1e+6, np.append(mtime, t4), t1, timex)

        t5, p5 = predict_t(ode_model, t0, t1, timex, dt, x0, mpres[-1] * (10 ** 6), [a, b], 0.5)
        cumtp4, cumgl4 = cumulator(np.append(mpres, p5) * (10 ** 6), x0, 0.1e+6, np.append(mtime, t5), t1, timex)

        ax.plot(t2, p2, 'c-', lw=0.25, alpha=0.2)
        ax.plot(t3,p3, 'g-',lw=0.25, alpha=0.3)
        ax.plot(t4, p4, 'm-', lw=0.25, alpha=0.4)
        ax.plot(t5,p5,'y-',lw=0.25, alpha=0.5)

        ax5.plot(cumtp1, np.array(cumgl1)+cumgl[-1], 'c-', lw = 0.25, alpha=0.2)
        ax5.plot(cumtp2,np.array(cumgl2)+cumgl[-1], 'g-',lw=0.25, alpha=0.3)
        ax5.plot(cumtp3, np.array(cumgl3)+cumgl[-1], 'm-', lw=0.25, alpha=0.4)
        ax5.plot(cumtp4,np.array(cumgl4)+cumgl[-1],'y-',lw=0.25, alpha=0.5)

        tm, pm = solve_ode(ode_model, 2009, 2019, 1 / 12, 25.16e+6, [a, b])
        ax.plot(t, pm, 'b-', lw=0.25, alpha=0.2)

    ax.plot([], [], 'k-', lw=0.5, alpha=0.4, label='model ensemble')

    # get the data

    ax.axvline(2019, color='b', linestyle=':', label='calibration/forecast')

    # 4. plot Wairakei data as error bars
    # *hint* see TASK 1 for appropriate plotting commands
    v = 2.
    ax.errorbar(tp, po, yerr=v, fmt='ro', label='data')
    ax.set_xlabel('time(years)')
    ax.set_ylabel('pressure(MPa)')
    ax.legend()

    mtime, mpres = solve_ode(ode_model, t0, t1, dt, x0, [7.5, 8e-6])
    tp1, pp1 = predict_t(ode_model, t0, t1, timex, dt, x0, mpres[-1] * (10 ** 6), [7.5, 8e-6], 2)
    cumtp1, cumgl1 = cumulator(np.append(mpres, pp1) * (10 ** 6), x0, 0.1e+6, np.append(mtime, tp1), t0, timex)

    tp2, pp2 = predict_t(ode_model, t0, t1, timex, dt, x0, mpres[-1] * (10 ** 6), [7.5, 8e-6], 1.5)
    cumtp2, cumgl2 = cumulator(np.append(mpres, pp2) * (10 ** 6), x0, 0.1e+6, np.append(mtime, tp2), t0, timex)

    tp3, pp3 = predict_t(ode_model, t0, t1, timex, dt, x0, mpres[-1] * (10 ** 6), [7.5, 8e-6], 1)
    cumtp3, cumgl3 = cumulator(np.append(mpres, pp3) * (10 ** 6), x0, 0.1e+6, np.append(mtime, tp3), t0, timex)

    tp4, pp4 = predict_t(ode_model, t0, t1, timex, dt, x0, mpres[-1] * (10 ** 6), [7.5, 8e-6], 0.5)
    cumtp4, cumgl4 = cumulator(np.append(mpres, pp4) * (10 ** 6), x0, 0.1e+6, np.append(mtime, tp4), t0, timex)

    ax5.plot(cumtp1, cumgl1, 'c-', label ='double')
    ax5.plot(cumtp2, cumgl2, 'g-', label ='1.5 times')
    ax5.plot(cumtp4, cumgl4, 'm-', label ='decrease')
    ax5.plot(cumtp3, cumgl3, 'y-', label='no change')
    ax5.plot(cumtime,cumgl, 'y-')
    ax5.legend()

    ax.plot(mtime, mpres, 'b-', label='model')
    ax.plot(tp1, pp1, 'c-', label='double')
    ax.plot(tp2,pp2, 'g-', label='1.5 times')
    ax.plot(tp3, pp3, 'm-', label = 'no change')
    ax.plot(tp4,pp4, 'y-', label ='decrease')
    ax.legend()
    ax.set_title('Uncertainty prediction of pressure')
    plt.show()



# Comment/uncomment each of the functions below as you complete the tasks

# TASK 1: Read the instructions in the function definition.
get_familiar_with_model()

# TASK 2: Read the instructions in the function definition.
a, b, posterior = grid_search()

# TASK 3: Open the file fun_with_multivariate_normals.py and complete the tasks.

# TASK 4: Read the instructions in the function definition.
# this task relies on the output of TASK 2, so don't comment that command
N = 200
samples = construct_samples(a, b, posterior, N)

# TASK 5: Read the instructions in the function definition.
# this task relies on the output of TASKS 2 and 3, so don't comment those commands
model_ensemble(samples)
