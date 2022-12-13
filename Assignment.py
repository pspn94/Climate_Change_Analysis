import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

#emma is crazy

def pandas_reader(path): 
    """The aim of this function is to r"""

    data_orig = pd.read_csv(path, skiprows = 4, index_col = ['Country Name', 'Indicator Name'])
    data_orig.drop(['Indicator Code', 'Country Code',  'Unnamed: 66'], 1, inplace = True)
    data = data_orig.reset_index()
    data_t = data_orig.transpose()
    return data, data_t

df, df_t = pandas_reader('API_19_DS2_en_csv_v2_4700503.csv')
print(df.head())
print('\n')
print(df_t.head())

def indicator_line_plot (tbl_name, indicator_value): 
    data = tbl_name[tbl_name['Indicator Name']== indicator_value] 
    data.set_index('Country Name', inplace = True)
    data.drop('Indicator Name', 1, inplace =True)
    data_1 = data.T
    data_1 = data_1.reset_index()
    data_1 = data_1.rename(columns = {'index': 'year'})
    data_1.plot(x ='year', y= ['Aruba', 'Africa Eastern and Southern', 'Afghanistan'], kind = 'line' )
    plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left')    
    plt.title('{}'.format(indicator_value))
    plt.show()
indicator_line_plot(df, *'Urban population'*)   
def bar_plot_indicator(tbl_name, indicator_value):
    data1 = tbl_name[tbl_name['Indicator Name']== indicator_value]
    data2 = data1.groupby(['Country Name'])['2017', '2018', '2019', '2020', '2021'].mean()[:10]
    data2.plot(kind = 'bar')
    plt.title('{} from 2018 till 2020'.format(indicator_value))
    
bar_plot_indicator(df, *'Urban population'*)
    
def country_correlation(tbl_name, country_name): 
    country = tbl_name[country_name]
    cols = []
    for x in tbl_name[country_name].columns:
        cols.append(x)
    tbl_1 = tbl_name[country_name]
    tbl = tbl_1[cols].iloc[:, :12]
    sns.heatmap(tbl.corr())
    plt.title('{}'.format(country_name))
    plt.legend([], frameon=False)
    
country_correlation(df_t, *'Brazil'*)