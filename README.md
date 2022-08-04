# Regression Model -  Birthweight Prediction Model - 

Built a predictive model on a continuous response variable to determine the explanatory variables – such as parents age, alcohol consumption, education – in relation with the dependent variable, the birthweight.

Dataset Dictionary
The dataset consist of 17 variables, and 196 observations

mage: mothers age
meduc: education in years
monpre: month prenatal care began
npvis: total number of prenatal visits
fage: father's age in years
feduc: father's educ in years
omaps: one minute apgar score, after event horizon
fmaps: five minute apgar score, after event horizon
cigs: ag cigarettes per day
drink: avg drinks per weeks
male: 1 if a baby male
mwhte: 1 if mother white
mblck: 1 if mother black
moth: 1 if mother is other
fwhte: 1 if father white
fblck: 1 if father is black
foth: if father is other
bwght: birthweight, grams
#importing necessary library
import pandas as pd
import numpy as np # 
import matplotlib.pyplot as plt #vizualization package
import seaborn as sns #vizualization package
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression
import sklearn.linear_model # linear models

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# specifying file name
file = './birthweight_low.xlsx'

# reading the file into Python
birthweight = pd.read_excel(file)

# outputting the first ten rows of the dataset
birthweight.head(n= 10)
mage	meduc	monpre	npvis	fage	feduc	omaps	fmaps	cigs	drink	male	mwhte	mblck	moth	fwhte	fblck	foth	bwght
0	69	NaN	5	2.0	62	NaN	4	7	23	9	1	0	1	0	0	1	0	697
1	68	12.0	3	10.0	61	11.0	4	6	25	11	1	1	0	0	1	0	0	1290
2	71	12.0	3	6.0	46	12.0	2	7	21	12	1	0	1	0	0	1	0	1490
3	59	16.0	1	8.0	48	16.0	7	8	21	10	0	0	0	1	0	0	1	1720
4	48	12.0	4	6.0	39	12.0	2	9	17	13	0	1	0	0	1	0	0	1956
5	67	11.0	4	8.0	40	8.0	4	9	16	14	0	1	0	0	1	0	0	1984
6	54	12.0	2	12.0	46	12.0	9	9	17	12	1	0	1	0	0	1	0	2050
7	71	14.0	4	7.0	51	11.0	9	8	15	13	0	1	0	0	1	0	0	2068
8	56	12.0	1	9.0	53	14.0	8	9	14	9	1	1	0	0	1	0	0	2148
9	58	12.0	2	12.0	61	16.0	9	9	13	6	0	0	1	0	0	1	0	2180
######  Analyzing dataframe quality
birthweight.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 196 entries, 0 to 195
Data columns (total 18 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   mage    196 non-null    int64  
 1   meduc   193 non-null    float64
 2   monpre  196 non-null    int64  
 3   npvis   193 non-null    float64
 4   fage    196 non-null    int64  
 5   feduc   189 non-null    float64
 6   omaps   196 non-null    int64  
 7   fmaps   196 non-null    int64  
 8   cigs    196 non-null    int64  
 9   drink   196 non-null    int64  
 10  male    196 non-null    int64  
 11  mwhte   196 non-null    int64  
 12  mblck   196 non-null    int64  
 13  moth    196 non-null    int64  
 14  fwhte   196 non-null    int64  
 15  fblck   196 non-null    int64  
 16  foth    196 non-null    int64  
 17  bwght   196 non-null    int64  
dtypes: float64(3), int64(15)
memory usage: 27.7 KB
Dataset if made up of 196 observation and 17 variables.Variables meduc, npvis and feduc contain missing values. At the same time we have three variables with a float data type. First, apply descriptive statistics to understand the data to gather additional insights on the data quality, and how to deal with the missing values.

Descriptive Statistics
#descriptive statistice on birthweight
birthweight.describe()
mage	meduc	monpre	npvis	fage	feduc	omaps	fmaps	cigs	drink	male	mwhte	mblck	moth	fwhte	fblck	foth	bwght
count	196.000000	193.000000	196.000000	193.000000	196.000000	189.000000	196.000000	196.000000	196.000000	196.000000	196.000000	196.000000	196.000000	196.000000	196.000000	196.000000	196.000000	196.000000
mean	40.153061	13.911917	2.341837	11.601036	39.290816	13.846561	8.193878	8.964286	10.928571	5.397959	0.551020	0.270408	0.382653	0.346939	0.346939	0.341837	0.311224	3334.086735
std	10.250055	2.055864	1.355136	4.267293	8.982725	2.634217	1.576482	0.651428	6.101282	3.001674	0.498664	0.445308	0.487279	0.477215	0.477215	0.475540	0.464180	646.700904
min	23.000000	8.000000	1.000000	2.000000	23.000000	1.000000	2.000000	5.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	697.000000
25%	33.000000	12.000000	2.000000	10.000000	34.750000	12.000000	8.000000	9.000000	6.000000	4.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	2916.250000
50%	39.000000	14.000000	2.000000	12.000000	38.000000	14.000000	9.000000	9.000000	11.000000	5.000000	1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	3452.000000
75%	46.000000	16.000000	3.000000	12.000000	43.000000	16.000000	9.000000	9.000000	15.250000	7.250000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	3759.500000
max	71.000000	17.000000	8.000000	35.000000	73.000000	17.000000	10.000000	10.000000	25.000000	14.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	4933.000000
Mother's Age (mage): maternal age range from 23 to 71 years old. On average the maternal age is 40 years old, with a standard deviation of 10 years. The data represents a wide rage for maternal ages.Advanced maternal age,mother being aged 35 years or above at the time of birth, is associated with increased risk of low birth weight and preterm delivery. As mentioned before, data shows the average maternal age is 40, therefore most of mothers are above the treshold for increased pregnacy risk, one being low birth weight.

Mother's Education (meduc): maternal educaton ranges from 8 - 17 years of education. During 8th grade, students are 13–14 years old in this stage of education. In other words is possible that our data shows teen pregnacies, which is a important factor in predicting birthweight. However futher analysis will need to be done to determine its relevance. On the other, on average mothers have completed 13.9 years of education.

Month Prenatal Care Starts (monpre):On average mothers start their prenatal care around the second month of their pregnacy. Prenatal care ranges from the 2 to 8 moths. According to the American Pregnancy Association, most women find out they're pregnant during their fourth to seventh week of pregnancy, meaning that they start prenatal careas soon as they find out they are pregnant. Early and comprehensive prenatal care is a significant indicators birthweight. Mothers who do not receive adequate maternity care, double the risk of. Futher analyis will be done to understand how the data between prenatal care and birthweight

-

Missing Values
For mother's education (meduc) the data is slightly skeweed to the left - as shown on the histogram below "Mothers Age Distribution", therefore median is a better representative of mother's education.

For fathers education (feduc) , the mean is pulled by extreme high values, therefore the median is a better representative of the data

For the number of prenatal visist (npvis), the mean seems to be pulled by extreme low values. This is show on the histogram below "#Prenatal Care Visits Distribution", where the data is skeweed to the left. At the same time the median and the third quartile are the same. Where the media says that approximatly 50% of mothers have a 12 year prenatal care visist during their pregnacy and due to the skweenes approximately 75% of mothers also fall on 12 prenatal care visits during their pregnacy.

These findings aling with what is expected on the real world. Mothers typically have seven to 12 prenatal visits over the course of a normal pregnancy (Price, 2015). Therefore the median is a better representative of npvis.

#Histogram to undestand the distibution of variables with missing values
#Mothers Age Distributin
fig, ax = plt.subplots(figsize = (8, 6))
plt.title("Mothers Age Distributin")
plt.show(sns.histplot(data=birthweight, x= 'meduc'))
plt.show()

#Fathers education distribution
fig, ax = plt.subplots(figsize = (8, 6))
plt.title("Father's Education Distribution")
plt.show(sns.histplot(data=birthweight, x= 'feduc'))
plt.show()


#Prenatal  Care Visits Distribution
fig, ax = plt.subplots(figsize = (8, 6))
plt.title("#Prenatal  Care Visits Distribution")
plt.show(sns.histplot(data=birthweight, x= 'npvis'))
plt.show()



# creating columns with 1s if missing and 0 if not
for colums in birthweight:

    # creating columns with 1s if missing and 0 if not
    if birthweight[colums].isnull().sum() > 0:
        birthweight['m_'+colums] = birthweight[colums].isnull().astype(int)

# summing the missing value flags to check the results of the loop above
print(birthweight[    ['m_meduc', 'm_npvis',
             'm_feduc']    ].sum(axis = 0).astype(int))
m_meduc    3
m_npvis    3
m_feduc    7
dtype: int32
#checking percentage of missing values to total datatset to determine relevance
birthweight['mv_sum']= birthweight['m_meduc'] + \
                     birthweight['m_npvis'] + \
                     birthweight['m_feduc']
                    
print(f"""

Number of Missing Values per Observation (Pct)
----------------------------------------------
{(birthweight['mv_sum'].value_counts(normalize = True,
                                sort = True,
                                ascending = True)*100).round(0)}
""")
Number of Missing Values per Observation (Pct)
----------------------------------------------
2     2.0
1     4.0
0    95.0
Name: mv_sum, dtype: float64

In totality 6% of data has missing values.Where 2% of the data has observations with 2 missing values while 4% percent of the data has observations with a least 1 missing value.

Filling Missing Values
#defining the median foe meduc, feduc, npvis
meduc_median = birthweight['meduc'].median()
feduc_median = birthweight['feduc'].median()
npvis_median  = birthweight['npvis'].median()

# filling medu, feduc and npvis missing values with the MEDIAN
birthweight['meduc'].fillna(value = meduc_median,
                         inplace = True)
birthweight['feduc'].fillna(value = feduc_median,
                         inplace = True)
birthweight['npvis'].fillna(value = npvis_median,
                            inplace = True)
# checking to make sure NAs are filled in
print(f"""
{'_' * 40}

Any missing values for mother, father & npvis?
{'_' * 40}

{birthweight['meduc'].isnull().any()}
{birthweight['feduc'].isnull().any()}
{birthweight['npvis'].isnull().any()}
""")
________________________________________

Any missing values for mother, father & npvis?
________________________________________

False
False
False

Converting Data Types
#converting floats to interger
birthweight ['meduc'] = birthweight ['meduc'].astype(int)
birthweight ['npvis'] = birthweight ['npvis'].astype(int)
birthweight ['feduc'] = birthweight ['feduc'].astype(int)
#checking results
birthweight.head(n=5)
mage	meduc	monpre	npvis	fage	feduc	omaps	fmaps	cigs	drink	male	mwhte	mblck	moth	fwhte	fblck	foth	bwght	m_meduc	m_npvis	m_feduc	mv_sum
0	69	14	5	2	62	14	4	7	23	9	1	0	1	0	0	1	0	697	1	0	1	2
1	68	12	3	10	61	11	4	6	25	11	1	1	0	0	1	0	0	1290	0	0	0	0
2	71	12	3	6	46	12	2	7	21	12	1	0	1	0	0	1	0	1490	0	0	0	0
3	59	16	1	8	48	16	7	8	21	10	0	0	0	1	0	0	1	1720	0	0	0	0
4	48	12	4	6	39	12	2	9	17	13	0	1	0	0	1	0	0	1956	0	0	0	0
omaps and fmaps variables, happens after the event horizon, therefore they will not be used for futher analysis.

Exploratory Analysis
Birthweigth Distribution
# rendering the plot
fig, ax = plt.subplots(figsize = (8, 6))
plt.show(sns.histplot(data=birthweight, x= 'bwght'))

Gender
# printing columns percetages
print(f"""
Baby is Male
------                  
{birthweight['male'].value_counts()}  

Baby is Male %
-----------
{round(birthweight['male'].value_counts()/len(birthweight)*100,0)}
""")
Baby is Male
------                  
1    108
0     88
Name: male, dtype: int64  

Baby is Male %
-----------
1    55.0
0    45.0
Name: male, dtype: float64

#rendereing plot for male
sns.boxplot(data=birthweight, x = 'male', y = 'bwght')
# title and axis labels 
plt.title(label   = "Birthweight Distribution by Gender")
plt.xlabel(xlabel = "Is Maler") 
plt.ylabel(ylabel = "Birthweight")
plt.show()

Analysis of Mother's Relationship to Birthweight
Maternal Age
#checking the distribution o for mothers age
# rendering the plot
fig, ax = plt.subplots(figsize = (8, 6))
plt.title("Mother's Age Distribution")
plt.xlabel("Age")
plt.show(sns.histplot(data=birthweight, x= 'mage'))

# Analysing relationship between birthweight and maternal age

#birthweight relationship to maternal age
plt.scatter(data=birthweight,
            x = "mage",
            y = "bwght",
           )
#customizing plot
plt.xlabel("Maternal Age")
plt.ylabel("Birthweight")
plt.title("Birthweight Relationship to Maternal Age")

#rendering plot
plt.show()

Maternal Education
#checking the distribution o for mothers eduction
# rendering the plot
fig, ax = plt.subplots(figsize = (8, 6))
plt.title("Mother's Education Distribution")
plt.xlabel("Educationi in years")
plt.show(sns.histplot(data=birthweight, x= 'meduc'))

# Analysing relationship between birthweight and maternal education

#birthweight relationship to maternal age
sns.boxplot(data=birthweight,
            x = "meduc",
            y = "bwght",
           )
#customizing plot
plt.xlabel("Maternal Education")
plt.ylabel("Birthweight")
plt.title("Birthweight Relationship to Maternal Education")

#rendering plot
plt.show()

Mother's Race
#Variables Value Count and Percentages for mwhte,  and mblck
print(f"""
Mother is White
----------
{birthweight['mwhte'].value_counts()}

Mother is White %
-----------
{round(birthweight['mwhte'].value_counts()/len(birthweight)*100,0)}

Mother is Black 
-------------
{birthweight['mblck'].value_counts()}

Mother is Black %
-----------
{round(birthweight['mblck'].value_counts()/len(birthweight)*100,0)}
""")
Mother is White
----------
0    143
1     53
Name: mwhte, dtype: int64

Mother is White %
-----------
0    73.0
1    27.0
Name: mwhte, dtype: float64

Mother is Black 
-------------
0    121
1     75
Name: mblck, dtype: int64

Mother is Black %
-----------
0    62.0
1    38.0
Name: mblck, dtype: float64

#displaying birthweigth against mother's race
sns.boxplot(data= birthweight,
            x= 'mblck',
            y= 'bwght',)
# title and axis labels
plt.title(label   = "Birthweight Relationship to Mother Race when white")
plt.xlabel(xlabel = "Mother is White")  #avoiding using dataset labels
plt.ylabel(ylabel = "Birthweigth")

# displaying the histogram
plt.show()

sns.boxplot(data= birthweight,
            x= 'mwhte',
            y= 'bwght',)

# title and axis labels
plt.title(label   = "Birthweight Relationship to Mothers Race when Black")
plt.xlabel(xlabel = "Mother is Black")  #avoiding using dataset labels
plt.ylabel(ylabel = "Birthweigth")

# displaying the histogram
plt.show()


Mother is other
#Variables Value Count and Percentages moth

print(f"""
Mother is other
------
{birthweight['moth'].value_counts()}

Mother is other %
-----------
{round(birthweight['moth'].value_counts()/len(birthweight)*100,0)}

""")
Mother is other
------
0    128
1     68
Name: moth, dtype: int64

Mother is other %
-----------
0    65.0
1    35.0
Name: moth, dtype: float64


#displaying birthweigth when mother is other
sns.boxplot(data= birthweight,
            x= 'moth',
            y= 'bwght',)
# title and axis labels
plt.title(label   = "Birthweigth relationship when mother is other)")
plt.xlabel(xlabel = "Mother is other")  #avoiding using dataset labels
plt.ylabel(ylabel = "Birthweigth")

# displaying the histogram
plt.show()

Prenatal Care
#checking the distribution o for number of prenatal care visist
# rendering the plot
fig, ax = plt.subplots(figsize = (8, 6))
plt.title("Prenatal Care Visits Distribution")
plt.xlabel("Number of Visits")
plt.show(sns.histplot(data=birthweight, x= 'npvis'))

#Analysing relationship between birthweight and when the number of prenatal care visist
sns.boxplot(data= birthweight,
            x= 'npvis',
            y= 'bwght',)
# title and axis labels
plt.title(label   = "Birthweigth relationship to number of prenatal care visist)")
plt.xlabel(xlabel = "Number of Visits")  #avoiding using dataset labels
plt.ylabel(ylabel = "Birthweigth")

# displaying the histogram
plt.show()

The boxplot show that around 12 prenatal care visits the baby is around 3400 grams, and as the number of visits decreases, so those the birweight. this related to complication during the pregnacy that would warrant higher number of visits, conseqiely indicating that the birthweigh is also beign affected byt said complications. However is really unclear the median birthweight of the baby drops around 10 prenatal visits

#checking the distribution o for start of prenatal care month
# rendering the plot
fig, ax = plt.subplots(figsize = (8, 6))
plt.title("Start Month of Prenatal Care Distribution")
plt.xlabel("Month")
plt.show(sns.histplot(data=birthweight, x= 'monpre'))

#Analysing relationship between birthweight and when the prenatal care began
sns.boxplot(data= birthweight,
            x= 'monpre',
            y= 'bwght',)
# title and axis labels
plt.title(label   = "Birthweigth relationship to when prenatal care began")
plt.xlabel(xlabel = "Month")  #avoiding using dataset labels
plt.ylabel(ylabel = "Birthweigth")

# displaying the histogram
plt.show()

Fathers and the relationship with birthweight
Father's Age
#checking the distribution of features related to the father age and education
# rendering the plot for father's age
fig, ax = plt.subplots(figsize = (8, 6))
plt.show(sns.histplot(data=birthweight, x= "fage"))
# rendering the plot for father's educ
fig, ax = plt.subplots(figsize = (8, 6))
plt.show(sns.histplot(data=birthweight, x= "feduc"))


Father's Education
#displayin birthweight against variables relating to father education and age.
plt.scatter(data= birthweight,
            x= 'fage',
            y= 'bwght',)

# title and axis labels
plt.title(label   = "Birthweight Distribution per Fathers's age")
plt.xlabel(xlabel = "Fathers age")  #avoiding using dataset labels
plt.ylabel(ylabel = "Birthweigth")
# displaying the histogram
plt.show()

sns.boxplot(data= birthweight,
            x= 'feduc',
            y= 'bwght',
            )
# title and axis labels
plt.title(label   = "Birthweight Distribution per Fathers's Education")
plt.xlabel(xlabel = "Father's Education") 
plt.ylabel(ylabel = "Birthweight")

# displaying the histogram
plt.show()


Father's Race
# printing columns percetages for fwhte, fblck, and foth
print(f"""
Father is White
----------
{birthweight['fwhte'].value_counts()} 

Father is White %
-----------
{round(birthweight['fwhte'].value_counts()/len(birthweight)*100,0)}

Father is Black
----------
{birthweight['fblck'].value_counts()} 

Father is Black %
-----------
{round(birthweight['fblck'].value_counts()/len(birthweight)*100,0)}

""")
Father is White
----------
0    128
1     68
Name: fwhte, dtype: int64 

Father is White %
-----------
0    65.0
1    35.0
Name: fwhte, dtype: float64

Father is Black
----------
0    129
1     67
Name: fblck, dtype: int64 

Father is Black %
-----------
0    66.0
1    34.0
Name: fblck, dtype: float64


#displaying birthweigth against fathers's race
sns.boxplot(data= birthweight,
            x= 'fwhte',
            y= 'bwght',)
# title and axis labels
plt.title(label   = "Birth Distribution per Father Race (white)")
plt.xlabel(xlabel = "Father is White")  #avoiding using dataset labels
plt.ylabel(ylabel = "Birthweigth")

# displaying the histogram
plt.show()

sns.boxplot(data= birthweight,
            x= 'fblck',
            y= 'bwght',)

# title and axis labels
plt.title(label   = "Birth Distribution per Father Race (black)")
plt.xlabel(xlabel = "Father is Black")  #avoiding using dataset labels
plt.ylabel(ylabel = "Birthweigth")

# displaying the histogram
plt.show()


Father is Other
# printing columns percetages for fwhte, fblck, and foth
print(f"""
Father is other
-------------
{birthweight['foth'].value_counts()}

Father is other %
-----------
{round(birthweight['foth'].value_counts()/len(birthweight)*100,0)}

""")
Father is other
-------------
0    135
1     61
Name: foth, dtype: int64

Father is other %
-----------
0    69.0
1    31.0
Name: foth, dtype: float64


#displaying birthweigth against mother's race
sns.boxplot(data= birthweight,
            x= 'foth',
            y= 'bwght',)
# title and axis labels
plt.title(label   = "Birth Distribution when Father is other)")
plt.xlabel(xlabel = "Father is other")  #avoiding using dataset labels
plt.ylabel(ylabel = "Birthweigth")

# displaying the histogram
plt.show()

Alchohol and Smoking Comsumptioin
# rendering the histogram alchohol and cigarret comsumption
fig, ax = plt.subplots(figsize = (8, 6))
plt.show(sns.histplot(data=birthweight, x= 'cigs'))

fig, ax = plt.subplots(figsize = (8, 6))
plt.show(sns.histplot(data=birthweight, x= 'drink'))


#rendering plot for cigs and alchohol comsumption
plt.scatter(data=birthweight, x= 'cigs', y='bwght')
# title and axis labels
plt.title(label   = "Birthweight Distribution vs Average Cigarrets per Day Comsumption")
plt.xlabel(xlabel = "Average Cigarrets per Day") 
plt.ylabel(ylabel = "Birthweight")
# displaying the histogram
plt.show()

#rendering plot for drinks
plt.scatter(data=birthweight, x= 'drink', y='bwght')
# title and axis labels
plt.title(label   = "Birthweight Distribution vs Average Drink per Week Comsumption")
plt.xlabel(xlabel = "Average Drinks per Week") 
plt.ylabel(ylabel = "Birthweight")
# displaying the histogram
plt.show()


Log Transformation
# log transforming mage and fage and saving it to the dataset
birthweight['log_mage'] = np.log(birthweight['mage'])
birthweight['log_fage'] = np.log(birthweight['fage'])
# comparing log transfomation distribution, histogram  
sns.histplot(data   = birthweight,
             x      = 'mage',
             kde    = True)


# rendering the plot
plt.show()

# histogram for log_mage
sns.histplot(data   = birthweight,
             x      = 'log_mage',
             kde    = True)


# rendering the plot
plt.show()

# histogram for fage 
sns.histplot(data   = birthweight,
             x      = 'fage',
             kde    = True)


# rendering the plot
plt.show()

# histogram for log_fage
sns.histplot(data   = birthweight,
             x      = 'log_fage',
             kde    = True)


# rendering the plot
plt.show()




# log transforming bwght and saving it to the dataset
birthweight['log_bwght'] = np.log(birthweight['bwght'])
#checking bwght distribution using a histogram
sns.histplot(data   = birthweight,
             x      = 'bwght',
             kde    = True)


# title and axis labels
plt.title(label   = "Original Distribution of Birth Weight")
plt.xlabel(xlabel = "bwght") # avoiding using dataset labels
plt.ylabel(ylabel = "Count")

# displaying the histogram
plt.show()
 
#checking log bwght distribution using a histogram    
sns.histplot(data   = birthweight,
             x      = 'log_bwght',
             kde    = True)


# title and axis labels
plt.title(label   = "Log Distribution of Birth Weight")
plt.xlabel(xlabel = "bwght") 
plt.ylabel(ylabel = "Count")

# displaying the histogram
plt.show()


Regression OLS
Perason Correlation
# creating a (Pearson) correlation matrix
birthweight_corr = birthweight.corr(method = 'pearson').round(2)


# printing (Pearson) correlations with bwght
print(birthweight_corr.loc['bwght'].sort_values(ascending = False))
bwght        1.00
log_bwght    0.97
fmaps        0.25
omaps        0.25
feduc        0.13
mblck        0.13
fblck        0.12
male         0.11
meduc        0.09
npvis        0.06
m_npvis      0.06
m_feduc     -0.00
moth        -0.02
mv_sum      -0.03
fwhte       -0.04
monpre      -0.05
foth        -0.08
mwhte       -0.11
m_meduc     -0.13
log_fage    -0.38
fage        -0.40
log_mage    -0.42
mage        -0.46
cigs        -0.57
drink       -0.74
Name: bwght, dtype: float64
# specifying plot size (making it bigger)
fig, ax = plt.subplots(figsize=(15,15))


# developing a coolwarm heatmap
sns.heatmap(data       = birthweight_corr, # the correlation matrix
            cmap       = 'coolwarm',    # changing to MEDIUM colors
            square     = True,          # tightening the layout
            annot      = True,          # should there be numbers in the heatmap
            linecolor  = 'black',       # lines between boxes
            linewidths = 0.5)           # how thick should the lines be


# title and displaying the plot
plt.title("""
Linear Correlation Heatmap for Birthweight Features
""")

plt.show()

# instantiating a scatter plot for mage and bwght
sns.lmplot(x          = 'mage',  # x-axis feature
           y          = 'bwght',  # y-axis feature
           hue        = None,     # categorical data for subsets
           legend_out = False,    # formats legend if hue != None
           scatter    = True,     # renders a scatter plot
           fit_reg    = True,    # renders a regression line
           aspect     = 2,        # aspect ratio for plot
           data       = birthweight) # DataFrame where features exist


# formatting and displaying the plot
plt.title(label    = "Mother's Age and Birthweight")
plt.xlabel(xlabel  = "Mother's Age")
plt.ylabel(ylabel  = 'Birthweigth')
# plt.xlim(20, 75)
plt.tight_layout()
plt.show()
C:\Users\Gleid\anaconda3\lib\site-packages\seaborn\regression.py:592: UserWarning: legend_out is deprecated from the `lmplot` function signature. Please update your code to pass it using `facet_kws`.
  warnings.warn(msg, UserWarning)

# instantiating a scatter plot for mage and bwght
sns.lmplot(x          = 'log_mage',  # x-axis feature
           y          = 'bwght',  # y-axis feature
           hue        = None,     # categorical data for subsets
           legend_out = False,    # formats legend if hue != None
           scatter    = True,     # renders a scatter plot
           fit_reg    = True,    # renders a regression line
           aspect     = 2,        # aspect ratio for plot
           data       = birthweight) # DataFrame where features exist


# formatting and displaying the plot
plt.title(label    = "Log Transformation - Mother's Age and Birthweight")
plt.xlabel(xlabel  = "Mother's Age")
plt.ylabel(ylabel  = 'Birthweigth')
plt.xlim(3, 4.5)
plt.tight_layout()
plt.show()
C:\Users\Gleid\anaconda3\lib\site-packages\seaborn\regression.py:592: UserWarning: legend_out is deprecated from the `lmplot` function signature. Please update your code to pass it using `facet_kws`.
  warnings.warn(msg, UserWarning)

Initial Thoughts
The linear trens seem to be consisten with lower age, howevent thre apppears to be a fannig out patter. This is indicative of a non linear relationship.
Birthweight seems to decline after mother 40 years old
Some observation greatly stand out from the linear trend, towards 70 years old thic oculd be that our data does not have alog of mother over the age 50
Let's try applying log_mage and log_bwght to see if it follos a liner relationship

# instantiating an lmplot for mage and bwght
sns.lmplot(x          = 'mage',  # x-axis feature
           y          = 'bwght',  # y-axis feature
           hue        = None,     # categorical data for subsets
           legend_out = False,    # formats legend if hue != None
           scatter    = True,     # renders a scatter plot
           fit_reg    = True,    # renders a regression line
           aspect     = 2,        # aspect ratio for plot
           data       = birthweight) # DataFrame where features exist


# looping over the x-range to save time
value = 20

while value < 73:
    
    # making a vertical line
    plt.axvline(x = value, color = "purple", linestyle = '--')
    
    # incrementing value in one-quarter increments
    value += 10


# formatting and displaying the plot
plt.title(label    = "Mother's Age and Birthweight")
plt.xlabel(xlabel  = "Mother's Age")
plt.ylabel(ylabel  = 'Birthweigth')
plt.xlim(20, 73)
plt.tight_layout()
plt.show()
C:\Users\Gleid\anaconda3\lib\site-packages\seaborn\regression.py:592: UserWarning: legend_out is deprecated from the `lmplot` function signature. Please update your code to pass it using `facet_kws`.
  warnings.warn(msg, UserWarning)

# creating a dummy column in the diamonds DataFrame
birthweight['Age size'] = 0


# for loop with iterrows() <-- one of the most useful methods for DataFrames
for index, col in birthweight.iterrows():
    
    
    # conditionals to change the values in the new column
    if birthweight.loc[index, 'mage'] < 30:
        birthweight.loc[index, 'Age size'] = '[20, 30)'
        
        
    elif birthweight.loc[index, 'mage'] < 40:
        birthweight.loc[index, 'Age size'] = '[30 - 40)'
        
        
    elif birthweight.loc[index, 'mage'] < 50:
        birthweight.loc[index, 'Age size'] = '[40 - 50)'
        
        
    elif birthweight.loc[index, 'mage'] < 60:
        birthweight.loc[index, 'Age size'] = '[50 - 50)'
        
    elif birthweight.loc[index, 'mage'] >= 70:
        birthweight.loc[index, 'Age size'] = '[70 - inf)'
    
    
    # safety net
    else:
        birthweight.loc[index, 'Age size'] = 'error'


# instantiating an lmplot for mage and bwght
sns.lmplot(x          = 'mage',  
           y          = 'bwght',  
           hue        = 'Age size', # categorical data for subsets
           legend     = False,        # supressing the legend
           legend_out = False,    
           scatter    = True,     
           fit_reg    = True,     
           aspect     = 2,        
           data       = birthweight)


# looping over the x-range to save time
value = 0

while value < 73:
    
    # making a vertical line
    plt.axvline(x = value, color = "purple", linestyle = '--')
    
    # incrementing value in 10 years increments
    value += 10


# formatting and displaying the plot
plt.title(label    = "Maternal Age and Birthweight")
plt.xlabel(xlabel  = 'Maternal Age')
plt.ylabel(ylabel  = 'Birthweight')
plt.xlim(20, 73)
plt.tight_layout()
plt.show()
C:\Users\Gleid\anaconda3\lib\site-packages\seaborn\regression.py:592: UserWarning: legend_out is deprecated from the `lmplot` function signature. Please update your code to pass it using `facet_kws`.
  warnings.warn(msg, UserWarning)

# instantiating an lmplot for log_mage and bwght
sns.lmplot(x          = 'log_mage',  # x-axis feature
           y          = 'bwght',  # y-axis feature
           hue        = None,     # categorical data for subsets
           legend_out = False,    # formats legend if hue != None
           scatter    = True,     # renders a scatter plot
           fit_reg    = True,    # renders a regression line
           aspect     = 2,        # aspect ratio for plot
           data       = birthweight) # DataFrame where features exist


# looping over the x-range to save time
value = 0

while value < 5:
    
    # making a vertical line
    plt.axvline(x = value, color = "purple", linestyle = '--')
    
    # incrementing value in 0.2 increments
    value += 0.2


# formatting and displaying the plot
plt.title(label    = " Log Transformation - Mother's Age and Birthweight")
plt.xlabel(xlabel  = "Mother's Age")
plt.ylabel(ylabel  = 'Birthweigth')
plt.xlim(3, 4.4)
plt.tight_layout()
plt.show()
C:\Users\Gleid\anaconda3\lib\site-packages\seaborn\regression.py:592: UserWarning: legend_out is deprecated from the `lmplot` function signature. Please update your code to pass it using `facet_kws`.
  warnings.warn(msg, UserWarning)

# creating a dummy column in the diamonds DataFrame
birthweight['Age size'] = 0


# for loop with iterrows() <-- one of the most useful methods for DataFrames
for index, col in birthweight.iterrows():
    
    
    # conditionals to change the values in the new column
    if birthweight.loc[index, 'log_mage'] < 3.2:
        birthweight.loc[index, 'Age size'] = '[3, 3.2)'
        
        
    elif birthweight.loc[index, 'log_mage'] < 3.4:
        birthweight.loc[index, 'Age size'] = '[3.2 - 3.4)'
        
        
    elif birthweight.loc[index, 'log_mage'] < 3.6:
        birthweight.loc[index, 'Age size'] = '[3.4.0 - 3.6)'
        
        
    elif birthweight.loc[index, 'log_mage'] < 3.8:
        birthweight.loc[index, 'Age size'] = '[3.6 - 3.8)'
        
        
    elif birthweight.loc[index, 'log_mage'] < 4.0: # this is where the bug was
        birthweight.loc[index, 'Age size'] = '[3.8 - 4.0)'
    
    elif birthweight.loc[index, 'log_mage'] >= 4.0:
        birthweight.loc[index, 'Age size'] = '[4.2 - inf)'
    
    
    # safety net
    else:
        birthweight.loc[index, 'Age size'] = 'error'


# instantiating an lmplot for log_mage and bwght
sns.lmplot(x          = 'log_mage',  
           y          = 'bwght',  
           hue        = 'Age size', 
           legend     = False,        # supressing the legend
           legend_out = False,    
           scatter    = True,     
           fit_reg    = True,     
           aspect     = 2,        
           data       = birthweight)


# looping over the x-range to save time
value = 0

while value < 5:
    
    # making a vertical line
    plt.axvline(x = value, color = "purple", linestyle = '--')
    
    # incrementing value in one-quarter increments
    value += 0.2


# formatting and displaying the plot
plt.title(label    = "Log Transformation - Maternal Age and Birthweight")
plt.xlabel(xlabel  = 'Maternal Age')
plt.ylabel(ylabel  = 'Birthweight')
plt.xlim(3.0, 4.4)
plt.tight_layout()
plt.show()
C:\Users\Gleid\anaconda3\lib\site-packages\seaborn\regression.py:592: UserWarning: legend_out is deprecated from the `lmplot` function signature. Please update your code to pass it using `facet_kws`.
  warnings.warn(msg, UserWarning)

# palplot for all numeric features
sns.pairplot(data = birthweight,
             x_vars = ['mage', 'log_mage', 'drink', 'cigs', 'bwght'],
             y_vars = ['mage', 'log_mage', 'drink', 'cigs', 'bwght'],
             hue = 'male',
             kind = 'scatter',
             palette = 'plasma')


# formatting options and display
plt.tight_layout()
plt.show()

#Creating new dataset and dropping unwated variables
birthweight_new  = birthweight.drop(['m_npvis','m_feduc','m_meduc','mv_sum','Age size', 'log_fage'],
                                axis = 1)
Base Model
Base models is developed using a minimal set of highly-correlated features from the original dataset based on pearson correlation and quality of linear relationship.This will also simplify the interpretation of model results.

#INSTANTIATE a model object
lm_base = smf.ols(formula = """bwght ~
                                       drink+
                                       cigs+
                                       mage  
                                            """,
                  data   = birthweight_new)


# Step 2: FIT the data into the model object
results = lm_base.fit()


# Step 3: analyze the SUMMARY output
print(results.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  bwght   R-squared:                       0.705
Model:                            OLS   Adj. R-squared:                  0.700
Method:                 Least Squares   F-statistic:                     152.7
Date:                Wed, 03 Aug 2022   Prob (F-statistic):           1.31e-50
Time:                        22:13:23   Log-Likelihood:                -1426.6
No. Observations:                 196   AIC:                             2861.
Df Residuals:                     192   BIC:                             2874.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   4928.7856    106.672     46.205      0.000    4718.385    5139.186
drink       -117.8497      9.480    -12.431      0.000    -136.549     -99.151
cigs         -36.0934      4.455     -8.103      0.000     -44.879     -27.307
mage         -14.0488      2.632     -5.338      0.000     -19.240      -8.858
==============================================================================
Omnibus:                        5.060   Durbin-Watson:                   1.278
Prob(Omnibus):                  0.080   Jarque-Bera (JB):                6.361
Skew:                          -0.145   Prob(JB):                       0.0416
Kurtosis:                       3.834   Cond. No.                         182.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Best OLS Model
#INSTANTIATE a model object
lm_best = smf.ols(formula = """bwght ~ log_mage +
                                       drink +
                                       cigs 
                                        """,
                  data    = birthweight_new)


# Step 2: FIT the data into the model object
results = lm_best.fit()


# Step 3: analyze the SUMMARY output
print(results.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  bwght   R-squared:                       0.695
Model:                            OLS   Adj. R-squared:                  0.690
Method:                 Least Squares   F-statistic:                     145.9
Date:                Wed, 03 Aug 2022   Prob (F-statistic):           2.78e-49
Time:                        22:13:23   Log-Likelihood:                -1429.7
No. Observations:                 196   AIC:                             2867.
Df Residuals:                     192   BIC:                             2880.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   6270.6416    394.420     15.898      0.000    5492.689    7048.594
log_mage    -514.7490    110.801     -4.646      0.000    -733.292    -296.206
drink       -121.2121      9.535    -12.713      0.000    -140.018    -102.406
cigs         -36.3316      4.525     -8.029      0.000     -45.257     -27.406
==============================================================================
Omnibus:                        7.224   Durbin-Watson:                   1.275
Prob(Omnibus):                  0.027   Jarque-Bera (JB):               10.158
Skew:                          -0.208   Prob(JB):                      0.00622
Kurtosis:                       4.035   Cond. No.                         224.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
The coefficient for bwght can be interpreted as follows:

For every one unit increase in mother's age, we expect an decrease in birthweight of approximately 514.74 grams, all else equal.

For every one unit increase in drink, we expect an decrease in birthweight of approximately 121.2 grams, all else equal.

For every one unit increase in cigs, we expect an decrease in birthweight of approximately 36.3 grams, all else equal.

This expectation is based on a confidence interval, as can be observed from the '0.025' and '0.975' columns. Thus, our interpretation of mother's age, drink, and cigs's effect on price can be expanded as follows:

For every one unit increase in age, we expect an decrease in birthweight between 296.2 grams and 733.29 grams a. Overall, we expect such decrease to converge on a value of 514.74 grams, all else equal.

For every one unit increase in drink, we expect an decrease in birthweight between 102.4 grams and 140 grams a. Overall, we expect such decrease to converge on a value of 121.2 grams, all else equal.

For every one unit increase in cigs, we expect a decrease in birthweight between 27.40 grams and 45.25 grams a. Overall, we expect such decrease to converge on a value of 36.3 grams, all else equal.

Training and Testing
#preparing explanatory variable data
birthweight_data   = birthweight_new.drop(['bwght',
                          'log_bwght'],
                                axis = 1)


# preparing response variables
birthweight_target = birthweight_new.loc[ : , 'bwght']
log_birthweight_target = birthweight_new.loc[ : , 'log_bwght']



# preparing training and testing sets 
x_train, x_test, y_train, y_test = train_test_split(
            birthweight_data,
            birthweight_target,
            test_size = 0.25,
            random_state = 219)


# checking the shapes of the datasets
print(f"""
Training Data
-------------
X-side: {x_train.shape}
y-side: {y_train.shape}


Testing Data
------------
X-side: {x_test.shape}
y-side: {y_test.shape}
""")
Training Data
-------------
X-side: (147, 18)
y-side: (147,)


Testing Data
------------
X-side: (49, 18)
y-side: (49,)

# declaring set of x-variables
x_variables = ['log_mage','drink','cigs']


# looping to make x-variables suitable for statsmodels
for val in x_variables:
    print(f"{val} +")
log_mage +
drink +
cigs +
# applying modelin scikit-learn

# preparing x-variables from the OLS model
ols_data = birthweight_new.loc[:,x_variables]


# preparing response variable
birthweight_target = birthweight_new.loc[:, 'bwght']


#saved for futher analysis with lasso model
# ###############################################
# ## setting up more than one train-test split ##
# ###############################################
# # FULL X-dataset (normal Y)
# x_train_FULL, x_test_FULL, y_train_FULL, y_test_FULL = train_test_split(
#             birthweight_data,     # x-variables
#             birthweight_target,   # y-variable
#             test_size = 0.25,
#             random_state = 219)


# OLS p-value x-dataset (normal Y)
x_train_OLS, x_test_OLS, y_train_OLS, y_test_OLS = train_test_split(
            ols_data,         # x-variables
            birthweight_target,   # y-variable
            test_size = 0.25,
            random_state = 219)
### INSTANTIATING a model object
lr = LinearRegression()


# FITTING to the training data
lr_fit = lr.fit(x_train_OLS, y_train_OLS)


# PREDICTING on new data
lr_pred = lr_fit.predict(x_test_OLS)

# SCORING the results
print('OLS Training Score :', lr.score(x_train_OLS, y_train_OLS).round(4))  # using R-square
print('OLS Testing Score  :', lr.score(x_test_OLS, y_test_OLS).round(4)) # using R-square
lr_train_score = lr.score(x_train_OLS, y_train_OLS)
lr_test_score  = lr.score(x_test_OLS, y_test_OLS)

# displaying and saving the gap between training and testing
print('OLS Train-Test Gap :', abs(lr_train_score - lr_test_score).round(4))
lr_test_gap = abs(lr_train_score - lr_test_score).round(4)
OLS Training Score : 0.6999
OLS Testing Score  : 0.6459
OLS Train-Test Gap : 0.054
