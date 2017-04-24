# Predicting Churn Using Survival Analysis
Capstone-Project-January-2017


## Questions to be asked
What are the good predictors of user lifetimes? More importantly, how do these predictors influence them.
How to extend user lifetimes?

## Dataset
User database from XX Company.
100,000 users over 5 years.
Each row contains >10 features. Example of features: Platform, gender, email, ...

## Modeling Approach
The Aalen Additive model, which is patient-survival regression model, was used to predict the lifetime. The model fits the data to a hazard function in turn is used to predict the lifetime. A detailed description of this machine learning methodology can be found in Ch.3 of master's dissertation by Huilin Gao.
http://archimede.mat.ulaval.ca/theses/H-Cao_05.pdf

The hazard function is composed of features (known as hazards) which can influence the lifetime of each user in our use case.

λ(t) = b0(t) + b1(t)x1 +. . . +bN(t)xT, where b0(t) is the baseline hazard function and xn are the hazards and bn are the parameters.

To train the model, we only use the users who have churned. In the survival-model parlance, the users who have not churned are "censored," so they are removed from the training. This is done because we do not know the lifetimes of these users yet. After the model is trained, we will use it to predict the lifetimes of the "censored" users.

FAQ

Why not a multi-decision tree model? Because a patient-survival model is more interpretable.
Why not linear or logistic regression? Because the residuals are not uniform over time. This is turn to due to the fact that the number of users decreases as more of them churn over time.
Why not a proportional hazard model, such as the Cox proportional hazard model? Because this assumes the hazards are in at a fixed proportion with each other over time. There is no reason to believe that is true in our case.

## Data processing
Python pandas was used to impute and clean up data.
Data package "Lifelines" which contains the Aalen additive model was used to model and plot the data. The Lifelines package can be installed by typing pip install lifelines.
Sample Python codes to train the models by bootstrapping and plot the results can be found in the 'Code' folder. A Python interpreter is needed to run them.
-The general approach is to use the users who have churned (uncensored) to train the model, which is then used to predict the lifetime ('Total_years') for the unchurned (censored) users.

##Results
##Conclusions
