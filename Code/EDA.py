import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def plot_Kaplan_Meier_overall(user_dataset):
	'''Accepts a dataframe of user data.  Plots the overall Kaplan-Meier curve based of the lifetime of the users.  The active users ('censored') will be excluded from the plot.

	Parameters:
	user_dataset: Pandas dataframe which contain at least the columns 'Total-years' and 'censored'.  'Total_years' represents how many years the users have been active.  'censored' indicates whether a user is still active (True = active user).

	Output:
	A Kaplan-Meier plot.

	This function does not return anything.

	'''
	#This produces two data frames of the columns 'Total_years'
	#and 'censored.'  The former indicates how manay years a
	#user has donoted before she/he churned.  The latter indicates
	#whether the user is censored (not churned).  Only user who
	#has churned (not censored) are used because we don't know the
	#'Total_years' of users who have not churned yet.
	T = user_dataset['Total_years']
	C = user_dataset['censored']

	#Create KaplanMeierInstance
	kmf = KaplanMeierFitter()
	kmf.fit(T, C, label = 'Overall')

	#plot KM function
	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(111)
	kmf.plot(ax=ax)
	ax.set_xlabel('Years', size = 20)
	ax.set_ylabel('Surviving user population', size = 20)
	ax.set_xlim(0,40)
	ax.set_ylim(0, 1)
	ax.grid()
	ax.legend(loc = 'best', fontsize = 20)
	plt.show()
	return

def plot_Kaplan_Meier_feature(user_dataset):
    '''Accepts a dataframe of user data.  For each feature (column), it plots the Kaplan-Meier curves of the users based on whether the feature is true or false.  The active users ('censored') will be excluded from the plot.

    Parameters:
    user_dataset: Pandas dataframe which contain at least the columns 'Total-years' and 'censored'.  'Total_years' represents how many years the users have been active.  'censored' indicates whether a user is still active (True = active user).

    Output:
    Kaplan-Meier plot(s).

    This function does not return anything.
    '''
    T = user_dataset['Total_years']
    C = user_dataset['censored']
    features = list(user_dataset.columns)
    features.remove('Total_years')
    features.remove('censored')
    features.remove('Baseline')
    kmf = KaplanMeierFitter()
    for feature in features:
        Above_mean = user_dataset[feature] > user_dataset[user_dataset['censored'] == 0][feature].mean()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        kmf = KaplanMeierFitter()
        kmf.fit(T[Above_mean], C[Above_mean], label = feature + ': Yes or > mean')
        kmf.plot(ax=ax, linewidth = 2)
        kmf.fit(T[~Above_mean], C[~Above_mean], label = feature + ': No or < mean')
        kmf.plot(ax=ax, linewidth = 2)
        ax.set_xlabel('Years', size = 10)
        ax.set_ylabel('Surviving user population', size = 10)
        ax.set_xlim(0,40)
        ax.set_ylim(0, 1)
        ax.grid()
        ax.legend(loc = 'upper right', fontsize = 10)
        plt.show()

def clean_data()
    return df_final
