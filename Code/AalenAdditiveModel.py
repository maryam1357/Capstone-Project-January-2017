import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from lifelines import AalenAdditiveFitter

'''This module contains the basicz functions for
    1. Instantiating aalen object from the AalenAdditiveFitter class.
    2. Performing bootstrapping on the dataset.
    3. Using the trained bootstrap models to predict the lifetime ('days') of users.
    4. Using the trained bootstrap models to create the cummulative hazard functions for users.
    '''

def Aalen_model(df, l2 = 0.01, coeff_pen = 0.1, smooth_pen = 0.1):
    '''Invokes the Aalen Additive Fitter class to creat an instance that fits the regression model:

    hazard(t)  = b_0(t) + b_1(t)*x_1 + ... + b_N(t)*x_N
	i.e., the hazard rate is a linear function of the covariates.

    Parameters
    df: Pandas dataframe.  The y column must be called "days."  A column of Boolean values called
        "kac" to indicate which row of data is kac, as indicated by True or False or 1 or 0.
    coeff_pen = 0.1: Attach a L2 penalizer to the size of the coeffcients during regression. This improves
        stability of the estimates and controls for high correlation between covariates.  For example,
        this shrinks the absolute value of c_{i,t}. Recommended, even if a small value.
    Smoothing_penalizer = 0.1: Attach a L2 penalizer to difference between adjacent (over time) coefficents. For
        example, this shrinks the absolute value of c_{i,t} - c_{i,t+1}.

	Other built-in, unadjustable parameters:
    Intercept = False.  We suggest adding a column of 1 to model the baseline hazard.
    nn_cumulative_hazard = True:  In its True state, it forces the the negative hazard values to be zero

    Output: aaf instance fitted to df'''
    aaf = AalenAdditiveFitter(fit_intercept=False, coef_penalizer=coeff_pen, smoothing_penalizer=smooth_pen, nn_cumulative_hazard=True)
    aaf.fit(df, 'days', event_col='kac')
    return aaf

def Bootstrap(df, bootstrap_count = 100):
    '''Accepts a pandas dataframe with a column called 'days' which is y and
    a column called 'kac' which indicates which column of data is kac.

    Performs bootstrap fitting with the Aalen model on a dataset df.
    Draws n=number of rows samples from df with replacement. Trains a Aalen model by invoking the Aalen_model
    function and append it into the list.  Repeat bootstrap_count times.
    Returns a list of bootstrap_count of Aalen models each trained by a bootstrapped set of data.

    Parameters:
    df: A pandas dataframe with y = 'days' and kac data = 'kac' and the rest of the columns are the features.
    bootstrap_count:  Number of bootstrapped model to train and append to the AAF_list

    Output: A list of length bootstrap_count of trained Aalen models.
    '''

    AAF_list = []  #This is a list to store the bootstrapped Aalen models
    for bootstrap_number in range(bootstrap_count):
        print 'bootstrap number', bootstrap_number
        df_Bootstrap = resample(df)
        aaf = Aalen_model(df_Bootstrap)  #Calls the Aalen_model function to train a model aaf
        AAF_list.append(aaf)  #Appends aaf to AAF_list
    return AAF_list

def Aalen_predict_lifetimes(AAF_list, test_dataset):
    '''Accepts a list (AAF_list) containing m trained bootstrapped AAF models (trained by bootstrapping in the Bootstrap function) and a test data set (test_dataset).  Calculate the predicted lifetimes by each model for each user.   Outputs the mean and median of lifetime for each user.

    The overall scheme is to iterate through the m bootstrap models and generate m predictions and m hazard functions for each row of data.  The mean and median of the predictions are calculated and added back to the the dataset as new columns called 'Mean_Pred_days' and 'Med_Pred_days'

    Parameters:
    AAF_list: A list of trained models to make predictions based of the test_dataset
    test_dataset: A pandas dataframe with y column = 'days'

    Output:  test_dataset with two columns 'Mean_Pred_days' and 'Med_Pred_days' added.
    '''
    aaf_predict = []

    for i, aaf in enumerate(AAF_list):
        #This informs the use the progress of the iteration
        print 'Performing prediction on model number ', i, '.'
        #Generates predictions for the dataset and appends them to the aaf_predict list.
        aaf_predict.append(aaf.predict_expectation(test_dataset.astype(float)).values)

    #Calculation of mean and median of aaf_predict
    aaf_pred_array = np.asarray(aaf_predict)
    test_dataset['Mean_Pred_days'] = np.mean(aaf_pred_array, axis = 0)
    test_dataset['Med_Pred_days'] = np.median(aaf_pred_array, axis = 0)

    return test_dataset

def Aalen_cum_haz(AAF_list, test_dataset):
    '''Accepts a list (AAF_list) containing m trained bootstrapped AAF models (trained by bootstrapping in the Bootstrap function)
    and a test data set (test_dataset), iterates through the list, outputs the cumulative hazard function for each
    feature (hazard), and returns them in a list.

    Note that these dataframes cannot be simply averaged because the time intervals are not always the same.

    Parameters:
    AAF_list: A list of trained models to make predictions
    test_dataset: A pandas dataframe with y column = 'days'

    Output:
    aaf_cum_haz:  A list of dataframes each presenting the hazard functions of the models in AAF_list.
    '''

    aaf_cum_haz = []
    for i, aaf in enumerate(AAF_list):
    #This informs the use the progress of the iteration
        print 'Performing predictions for model ', i , '.'
        aaf_cum_haz.append(aaf.predict_cumulative_hazard(test_dataset))
    #Generates hazard functions that are tailored to each user and append them to aaf_predict_cum_haz

    plot_user_cum_haz(aaf_cum_haz)
    return aaf_cum_haz

def plot_cum_haz_functions(AAF_list, x_max =40):
	'''Accepts a list of Aalen_model instances and plot the cumulative hazard function for all instances.
	This function does not return anythng.
	Parameters:
	AAF_list: a list of Aalen additive instances created by the AalenAdditiveFitter
	x_max = 40: maxiumum value of the x-axis.

	Output:
	Plots of the individual hazard functions for each specific hazard as a function of time.
	'''

	colors = ['grey', 'lightgreen', 'blue','red', 'magenta', 'gold', 'green', 'orange']
	for i, hazard in enumerate(AAF_list[0].hazards_.columns):
		print 'hazard is', hazard
		fig = plt.figure(figsize=(5, 5))
		plt.subplot(111)

		#To avoid having a legend for each of the model (therefore leading having multiple labels of the
		#same data, we plot only the first model with legend here, and the rest of the model without legend)
		aaf = AAF_list[0]
		plt.plot(aaf.cumulative_hazards_[hazard], alpha =0.05, c = colors[i], label = hazard)
		plt.legend(loc = 'lower right', fontsize = 10)

		#These models are plotted without legend.
		for j in range(1, len(AAF_list)):
			aaf = AAF_list[j]
			plt.plot(aaf.cumulative_hazards_[hazard], alpha =0.05, c = colors[i], label = hazard)

		#Customize the axes, labels, etc.
		plt.xlabel('Years', size = 10)
		plt.ylabel('Cumulative hazard function', size = 10)
		plt.grid()
		plt.xlim(0, x_max)
		plt.show()
	plt.close

def plot_user_cum_haz(user_cum_haz_functions_list, number_of_users = 8, years = 5, y_max = 5, lw = 0.01):
    '''Accepts a list of (bootstrap) trained models, calculates each of their cumulative hazard functions,
	randomly selects number_of_users (kac), and plots their hazard functions.

    Parameters:
	'days' represents how many years the users have been active.
	'kac' indicates whether a user is still active (True = active user).

    Output:
    A plot of the cumulative hazard functions for the randomly selected users.

    This function does not return anything.
    '''

	#This list of colors will be cycled through during the plotting.  If the number_of_users > 8, the colors
	#will have to be repeated.
    colors = ['grey', 'lightgreen', 'blue','red', 'magenta', 'gold', 'green', 'orange']

	#Randomly selects a number_of_users from the user_dataset to be plotted.
    rows = random.sample(range(user_cum_haz_functions_list[0].shape[1]), number_of_users)

	#Iterates through the rows (users selected) and then through all cumulative hazard functions
	#to plot them on the same figure.  Note that this order of iteraton must be maintained
	#so that the cum hazard function for each user is plotted with the same color.
    for i, row in enumerate(rows):
		print row
		for user_cum_haz_functions in user_cum_haz_functions_list:
			plt.plot(user_cum_haz_functions.T.iloc[row], alpha = lw, color = colors[i%8])
    plt.xlabel('Years', size = 10)
    plt.ylabel("user cumulative hazard function", size = 10)
    text = 'Cumulative hazards for ' + str(number_of_users) + ' random kac users.'
    plt.annotate(text, xy=(0, 0), xytext=(0.2, 2.8))
    plt.grid()
    plt.xlim(0, years)
    plt.ylim(0, y_max)
    plt.show()
    plt.close()
