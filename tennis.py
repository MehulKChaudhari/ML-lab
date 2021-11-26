import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
tennis = pd.read_csv('tennis.csv')
outlook = tennis.groupby(['Outlook', 'Play']).size() 
temp = tennis.groupby(['Temperature', 'Play']).size() 
humidity = tennis.groupby(['Humidity', 'Play']).size() 
windy = tennis.groupby(['Wind', 'Play']).size()
play = tennis.Play.value_counts()


def bayestheorem():
    print('Posterior [P(c|x)] - Posterior probability of the target/class (c) given predictors (x)'),
    print('Prior [P(c)] - Prior probability of the class (target)'),
    print('Likelihood [P(x|c)] - Probability of the predictor (x) given the class/target (c)'), 
    print('Evidence [P(x)] - Prior probability of the predictor (x))')

def bayesposterior(prior, likelihood, evidence, string): 
    print('Prior=', prior),
    print('Likelihood=', likelihood), print('Evidence=', evidence),
    print('Equation =','(Prior*Likelihood)/Evidence') 
    print(string, (prior*likelihood)/evidence)

ct = pd.crosstab(tennis['Outlook'], tennis['Play'], margins = True) 
print(ct)
ct.columns = ["no","yes","rowtotal"]
ct.index= ["overcast","rainy","sunny","coltotal"] 
ct / ct.loc["coltotal","rowtotal"]
ct / ct.loc["coltotal"]
bayesposterior(prior = ct.iloc[1,1]/ct.iloc[3,1], likelihood = ct.iloc[3,1]/ct.iloc[3,2], evidence = ct.iloc[1,2]/ct.iloc[3,2],
string = 'Probability of Tennis given Rain =')
