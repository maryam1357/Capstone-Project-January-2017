import matplotlib.pyplot as plt
import lifelines as sa
from lifelines import KaplanMeierFitter
import pandas as pd

def plot_Kaplan_Meier_overall(df_final_km):
	'''Accepts a dataframe of user data.  Plots the overall Kaplan-Meier curve based of the lifetime of the users.  The active users ('censored') will be excluded from the plot.

	Parameters:
	df_final_km: Pandas dataframe which contain at least the columns 'Total-years' and 'censored'.  'Total_years' represents how many years the users have been active.  'censored' indicates whether a user is still active (True = active user).

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
	print df_final_km.head()
	T = df_final_km['days']
	C = df_final_km['kac']

	#Create KaplanMeierInstance
	kmf = KaplanMeierFitter()
	kmf.fit(T, C, label = 'Overall')

	#plot KM function
	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(111)
	kmf.plot(ax=ax)
	ax.set_xlabel('days', size = 20)
	ax.set_ylabel('Surviving user population', size = 20)
	ax.grid()
	ax.legend(loc = 'best', fontsize = 20)
	plt.savefig('kaplan_meier_baseline.png')
	plt.close()
	return

def plot_Kaplan_Meier_feature(df_final_km):
	'''Accepts a dataframe of user data.  For each feature (column), it plots the Kaplan-Meier curves of the users based on whether the feature is true or false.  The active users ('censored') will be excluded from the plot.

	Parameters:
	df_final_km: Pandas dataframe which contain at least the columns 'Total-years' and 'censored'.  'Total_years' represents how many years the users have been active.  'censored' indicates whether a user is still active (True = active user).

	Output:
	Kaplan-Meier plot(s).

	This function does not return anything.
	'''
	print df_final_km.head()
	df_final_km_dropped = df_final_km.drop(['rollup_count','unsubscribe_count','inbox_count','fresh_count','unknown_count'], axis=1)
	features = list(df_final_km_dropped.columns)
	features.remove('days')
	features.remove('kac')
	features.remove('count_initial_scan_subscriptions')
	features.remove('kac_quarter')
	for feature in features:
		ax = plt.subplot(111)
		for item in df_final_km_dropped[feature].unique():
			df_sub = df_final_km_dropped.loc[df_final_km_dropped[feature]==item]
			km = sa.KaplanMeierFitter()
			km.fit(durations=df_sub['days'], event_observed=df_sub['kac'], label=item)
			ax = km.plot(ax=ax, label=feature)
		plt.savefig('kaplan_meier_feature_{}.png'.format(feature))
		plt.close()

def assign_email_group(string):
    if string.split('.')[-1]=='edu':
        return 'edu'
    elif string.startswith('gmail'):
        return 'gmail'
    else:
        return 'others'

def clean_user_data(df_user):
	print "Cleaning user data"
	df_user['email_domain']=df_user['email'].str.split('@').str.get(1)
	df_user['domains'] = df_user.email_domain.apply(assign_email_group)
	df_clean_user = df_user.drop(['email_domain','email','user_referral_detail','user_referral_id','user_referral_value','disconnect_timestamp','slice_user_id','kac_history','goodness_score','country','region','city','metro_code'], axis=1)
	print df_clean_user.head()
	return df_clean_user

def clean_filter(df_filter):
	print "Cleaning filter"
	df_filter_type_count = df_filter.groupby(['user_id', 'filter_type_id'])['filter_type_id'].count().unstack().fillna(0)
	df_filter_type_count = df_filter_type_count.rename(columns={0:'rollup_count', 1:'unsubscribe_count', 2:'inbox_count', 3:'fresh_count', 5:'unknown_count'})
	print df_filter_type_count.head()
	return df_filter_type_count

def merge_and_clean_user_history_data(df_clean_user, df_history):
	print "Merging and cleaning user and history data"
	df_clean_histroy = pd.DataFrame(df_history.groupby('user_id')['disconnect_timestamp'].max())
	df_clean_user_history = df_clean_user.join(df_clean_histroy, on='user_id')
	#df_user_history['active_time'] = pd.to_datetime(df_user_history.kac_update_timestamp) - pd.to_datetime(df_user_history.disconnect_timestamp)
	#df_user_history['not_active'] = (df_user_history.active_time.astype(int)==0).astype(int)
	df_clean_user_history = df_clean_user_history[df_clean_user_history['kac'].isin([1,3])]
	df_clean_user_history['kac_update_timestamp'] = pd.to_datetime(df_clean_user_history['kac_update_timestamp'])
	df_clean_user_history['join_timestamp'] = pd.to_datetime(df_clean_user_history['join_timestamp'])
	df_clean_user_history['days'] = (df_clean_user_history.kac_update_timestamp - df_clean_user_history.join_timestamp)
	df_clean_user_history['days'] = df_clean_user_history.days.dt.days
	df_clean_user_history['kac_quarter'] = df_clean_user_history.kac_update_timestamp.dt.quarter
	df_clean_user_history.kac = df_clean_user_history.kac.map({3: 1, 1: 0})
	df_clean_user_history['intial_scan_count'] = pd.cut(df_clean_user_history.count_initial_scan_subscriptions, bins=[-1, 355, 999999], labels=['<mean', '>mean'])
	print df_clean_user_history.head()
	return df_clean_user_history

def create_final_set(df_clean_user_history, df_filter_type_count):
	print "Creating the final km data set"
	df_final_km = df_clean_user_history.join(df_filter_type_count, on='user_id')
	df_final_km[['rollup_count', 'unsubscribe_count', 'inbox_count', 'fresh_count', 'unknown_count']] = df_final_km[['rollup_count', 'unsubscribe_count', 'inbox_count', 'fresh_count', 'unknown_count']].fillna(0)
	df_final_km['rollup'] = pd.cut(df_final_km.rollup_count, bins=[-1, 20, 999999], labels=['<mean', '>mean'])
	df_final_km['unsubscribe'] = pd.cut(df_final_km.unsubscribe_count, bins=[-1, 105, 999999], labels=['<mean', '>mean'])
	df_final_km['inbox'] = pd.cut(df_final_km.inbox_count, bins=[-1, 45, 999999], labels=['<mean', '>mean'])
	df_final_km['fresh'] = pd.cut(df_final_km.fresh_count, bins=[-1, 185, 999999], labels=['<mean', '>mean'])
	df_final_km = df_final_km.drop(['join_timestamp','kac_update_timestamp','disconnect_timestamp'], axis=1)
	a = df_final_km['user_id']
	df_final_km.set_index(a,inplace=True)
	df_final_km.drop('user_id', axis=1, inplace=True)
	print df_final_km.head()
	return df_final_km

def clean_data_km(userData, historyData, filterData):
	print "Cleaning km data set"
	df_user = pd.read_csv(userData, delimiter=';')
	df_history = pd.read_csv(historyData, delimiter=';')
	df_filter = pd.read_csv(filterData,delimiter=';')
	df_clean_user = clean_user_data(df_user)
	df_clean_user_history = merge_and_clean_user_history_data(df_clean_user, df_history)
	df_filter_type_count = clean_filter(df_filter)
	return create_final_set(df_clean_user_history, df_filter_type_count)

def clean_data(userData, historyData, filterData):
	print "Cleaning data"
	df_final_km = clean_data_km(userData, historyData, filterData)
	dummies_signup_origin = pd.get_dummies(df_final_km['domains']).rename(columns = lambda x: 'domains_'+str(x))
	df_final_km = pd.concat([df_final_km,dummies_signup_origin],axis=1)

	dummies_signup_origin = pd.get_dummies(df_final_km['signup_origin']).rename(columns = lambda x: 'signup_origin_'+str(x))
	df_final_km = pd.concat([df_final_km,dummies_signup_origin],axis=1)

       #u'count_initial_scan_subscriptions', u'not_active', u'days',
       #u'join_quarter', u'kac_quarter', u'f_type_0', u'f_type_1', u'f_type_2',
       #u'f_type_3', u'f_type_5', u'domains_gmail', u'domains_others',
       #u'signup_origin_Web App', u'signup_origin_iPhone'
	df_final_km = df_final_km.drop('signup_origin_Old Website', 1)
	df_final_aa = df_final_km.drop(['domains_edu','unknown_count','fresh_count','inbox_count','unsubscribe_count','rollup_count','domains_others','signup_origin','domains','rollup', 'unsubscribe', 'inbox', 'fresh','intial_scan_count', 'kac_quarter'], axis=1)
	print "Cleaning data completed!!!"
	print df_final_aa.head()
	return df_final_aa
	#,,,,'signup_origin_iPhone','count_initial_scan_subscriptions','domains_gmail','signup_origin_Web App'
