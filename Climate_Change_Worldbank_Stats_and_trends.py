# -*- coding: utf-8 -*-
"""
7PAM2000-0901-2022 - Statistics and trends

Pedro Neto
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def create_dfs(path):
    """
    This function was created to produce dataframes, one of then with
    countries as columns and the another one with years as columns.
    
    Arg:
        path (str): write the filename or the file path of a Worldbank
        dataframe
        
    Return:
        A original and a transposed Worldbank dataframe.  
    """
    df = pd.read_csv(path, skiprows=4)
    df.drop(['Country Code', 'Indicator Code', 'Unnamed: 66'],
            axis=1, inplace=True)
    df_T = pd.read_csv(path, skiprows=4,
                       index_col=['Indicator Name', 'Country Name'])
    df_T.drop(['Country Code', 'Indicator Code', 'Unnamed: 66'],
              axis=1, inplace=True)
    df_T = df_T.T
    df_T.index = df_T.index.astype('int64')

    return df, df_T

# setting the file name to a variable
FILE_NAME = 'API_19_DS2_en_csv_v2_4700503.csv'
# creating two dataframes using the function
df, df_T = create_dfs(FILE_NAME)
# analysing the df
print(df.head())
print(df_T.head())
print(df['Indicator Name'].unique())
print(df.columns)
# creating a list of choosen indicators
indicators = ['CO2 emissions (kt)', 'Urban population',
              'Agricultural land (% of land area)',
  'Prevalence of underweight, weight for age (% of children under 5)',
              'Access to electricity (% of population)']
# creating a list of a range of years as strings
years_str = ['1990', '1991', '1992', '1993', '1994',
             '1995', '1996', '1997', '1998', '1999',
             '2000', '2001', '2002', '2003', '2004',
             '2005', '2006', '2007', '2008', '2009',
             '2010', '2011', '2012', '2013', '2014',
             '2015', '2016', '2017', '2018', '2019']
# creating a list of a range of years as integers
years_int = [eval(i) for i in years_str]
# creating a df from 1990 until 2019
df_1990_until_2019 = df_T.filter(items=years_int, axis=0)
df_1990_until_2019
# analysing countries that are the largest CO2 emitters
print(df_1990_until_2019['CO2 emissions (kt)'].sum()\
      .sort_values(ascending=False).to_dict())
# creating a list of choosen countries
countries = ['China', 'United States', 'India', 'Russian Federation',
             'Japan', 'Germany']
# Exploring statistics of few indicatores
for ind in indicators:
    print(ind, '\n')
    print(df_1990_until_2019.loc[:, (slice(None), countries)][ind].describe())
    print()
    print("Average: ", np.average(df_1990_until_2019\
                                     .loc[:, (slice(None), countries)][ind]))
    print()
    print("Std. deviations:", np.std(df_1990_until_2019\
                                     .loc[:, (slice(None), countries)][ind]))
    print()
    print ("Skewness \n", stats.skew(df_1990_until_2019\
                                     .loc[:, (slice(None), countries)][ind]))
    print()
    print ("Kurtosis \n", stats.kurtosis(df_1990_until_2019.loc\
                                         [:, (slice(None), countries)][ind]))
    print()
    
# calculating China's population increase
print((df[(df['Country Name']=='China') &
          (df['Indicator Name']=='Urban population')]['2019']) /\
          (df[(df['Country Name']=='China') &
          (df['Indicator Name']=='Urban population')]['1990']) * 100)
# calculating India's population increase
print((df[(df['Country Name']=='India') &
          (df['Indicator Name']=='Urban population')]['2019']) /\
          (df[(df['Country Name']=='India') &
          (df['Indicator Name']=='Urban population')]['1990']) * 100)
# creating a df of incomes
df_incomes = df[(df['Country Name']=='Low income') |
                (df['Country Name']=='Middle income') |
                (df['Country Name']=='High income')]
# creating a df with df incomes selecting access to electricity
df_elec_income_T = df_incomes[df_incomes['Indicator Name']==\
                              'Access to electricity (% of population)']\
    .groupby(['Country Name'])[years_str[10:-1]].sum().T
df_elec_income_T.index = df_elec_income_T.index.astype(int)
# creating a df with df incomes selecting access to electricity
df_prev_income_T = df_incomes[df_incomes['Indicator Name']==\
        'Prevalence of underweight, weight for age (% of children under 5)']\
    .groupby(['Country Name'])[years_str[10:-1]].sum().T
df_prev_income_T.index = df_elec_income_T.index.astype(int)   
    
# plotting CO2 emissions line graph
fig, ax = plt.subplots(dpi=240)

ax.plot(df_1990_until_2019.loc[:, (slice(None), countries[0])]\
        ['CO2 emissions (kt)'], label=countries[0], linewidth=2.5)
for con in countries[1:]:
    ax.plot(df_1990_until_2019.loc[:, (slice(None), con)]\
            ['CO2 emissions (kt)'], label=con, linewidth=1.5)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
    
plt.title('CO2 emissions (kt) (1990-2019)')
plt.xlabel('Years')
plt.ylabel('Emissions')

plt.tight_layout
plt.savefig('plot1', dpi=300);

# plotting Urban population line graph
fig, ax = plt.subplots(dpi=240)

ax.plot(df_1990_until_2019.loc[:, (slice(None), countries[0])]\
        ['Urban population'], label=countries[0], linewidth=2.5)
for con in countries[1:]:
    ax.plot(df_1990_until_2019.loc[:, (slice(None), con)]\
            ['Urban population'], label=con, linewidth=1.5)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
    
plt.title('Urban population (1990-2019)')
plt.xlabel('Years')
plt.ylabel('Urban population')

plt.tight_layout()
plt.savefig('plot2', dpi=300);

# plotting heatmap for each choosen country
for coun in countries:
    ind_names = ['CO2 emissions (kt)',
             'Urban populations',
             'Agricultural Land (%)',
             'Prevalence of underweight childrens',
             'Access to electricity']
    
    fig, ax = plt.subplots(dpi=240)

    sns.heatmap(df_1990_until_2019.loc[:, (slice(None), [coun])]\
                [indicators].corr(),
                fmt='.2f', square=True, linecolor='white', vmax=1.0,
                annot=True, xticklabels=False, yticklabels=ind_names)\
        .set(title=coun, xlabel='', ylabel='')
        
    plt.savefig(coun, dpi=300)
    plt.show();
    

# plotting heatmap of world
fig, ax = plt.subplots(dpi=240)
ind_names2 = ['Prevalence of underweight (%)', 'Access to electricity (%)']
sns.heatmap(df_1990_until_2019.loc[:, (slice(None), ['World'])]\
            [indicators[3:5]].corr(),
            fmt='.2f', square=True, linecolor='white', vmax=1.0,
            annot=True, xticklabels=False, yticklabels=ind_names2)\
    .set(title='World', xlabel='', ylabel='')

plt.savefig('plot3', dpi=300)
plt.show();

# plotting bar chart of incomes
fig, ax = plt.subplots(dpi=240)

years=19
r = np.arange(years)
width = 0.25

ax.bar(r, df_elec_income_T['Low income'],
       width=width, label='Low income')
ax.bar(r+width, df_elec_income_T['Middle income'],
       width=width, label='Middle income')
ax.bar(r+(2*width), df_elec_income_T['High income'],
       width=width, label='High income')

ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks(r+width, years_str[10:-1], rotation=90)
ax.set_yticks([])

plt.legend(bbox_to_anchor=(0., .97, 1., .102), loc='lower left',
                      ncol=3, mode="expand")
fig.suptitle('Access to electricity percentage (2000-2018)', fontsize=12)

plt.tight_layout()
plt.savefig('plot4', dpi=300);

# plotting bar chart of incomes
fig, ax = plt.subplots(dpi=240)

years=19
r = np.arange(years)
width = 0.25

ax.bar(r, df_prev_income_T['Low income'],
       width=width, label='Low income')
ax.bar(r+width, df_prev_income_T['Middle income'],
       width=width, label='Middle income')
ax.bar(r+(2*width), df_prev_income_T['High income'],
       width=width, label='High income')

ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks(r+width, years_str[10:-1], rotation=90)
ax.set_yticks([])

plt.legend(bbox_to_anchor=(0., .97, 1., .102), loc='lower left',
                      ncol=3, mode="expand")
fig.suptitle('Prevalence of underweight childrens (2000-2018)', fontsize=12)

plt.tight_layout()
plt.savefig('plot5', dpi=300);