import numpy as np
import pandas as pd
import scipy

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Extract the data. Return both the raw data and dataframe
def generateDataset(filename):
    data = pd.read_csv(filename)
    df = data[0:]
    df = df.dropna()
    return data, df

#Run a t-test
def runTTest(ivA, ivB, dv):
    ttest = scipy.stats.ttest_ind(ivA[dv], ivB[dv])
    print(ttest)

def runAnova(data, formula):
    model = ols(formula, data).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    print(aov_table)

def addPercentAdmitted(data):
    data['percent_admitted'] = data['Admitted']/(data['Admitted']+data['Rejected'])
    data['percent_rejected'] = data['Rejected']/(data['Admitted']+data['Rejected'])
    data.to_csv("simpsons_paradox_2.csv")


rawData, df = generateDataset("simpsons_paradox.csv")

print("Does gender correlate with admissions?")
men = df[df['Gender'] == "Male"]
women = df[df['Gender'] == "Female"]
runTTest(men, women, "Admitted")

print("Does department correlate with admissions?")
simpleFormula = "Admitted ~ C(Department)"
runAnova(rawData, simpleFormula)

print("Do gender and department correlate with admissions?")
moreComplex = 'Admitted ~ C(Department) + C(Gender)'
runAnova(rawData, moreComplex)

addPercentAdmitted(rawData)


#Problem set 11 -> Problem 1

#(a) -
#Independent var: current year (categorical)
#Dependent var: GPA (continuous)
#Statistical test: t-test

#(b)
#Independent var: time (continuous)
#Dependent var: amount of snow fall (continuous)
#Statistical test: generalized Regression

#(c)
#Independent var: season (categorical)
#Dependent var: number of hikers (continuous)
#Statistical test: t-test

#(d)
#Independent var: state (categorical)
#Dependent var: highest degree level (categorical)
#Statistical test: Chi-squared

#Problem set 11 -> Problem 2

#At first glance, when looking at the new variables, it seems that gender does have
#a significant influence on the percent of students admitted. The department also seems
#to have an effect on addmissions rate overall. By just looking at the data, it seems
#that admissions bias is definitely present.
