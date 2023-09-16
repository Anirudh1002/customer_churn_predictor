import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, option):

    #Defining the map function
    def binary_map(feature):
        return feature.map({'Male':1, 'Female':0})

    # Encode binary categorical features
    binary_list = ['Gender']
    df[binary_list] = df[binary_list].apply(binary_map)

    
    #Drop values based on operational options
    columns = ['Age','Gender', 'Subscription_Length_Months',	'Monthly_Bill',	'Total_Usage_GB', 'Location_Houston', 'Location_Los Angeles', 'Location_Miami', 'Location_New York']
    #Encoding the other categorical categoric features with more than two categories
    df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)



    #feature scaling
    sc = MinMaxScaler()
    df['Age'] = sc.fit_transform(df[['Age']])
    df['Subscription_Length_Months'] = sc.fit_transform(df[['Subscription_Length_Months']])
    df['Monthly_Bill'] = sc.fit_transform(df[['Monthly_Bill']])
    df['Total_Usage_GB'] = sc.fit_transform(df[['Total_Usage_GB']])
    return df
        




