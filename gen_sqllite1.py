# Requires xlrd for reading Excel files

import pandas as pd
import sqlite3
import datetime

def get_input_df(filename, cats_filename):
    # Read input from CSV or Excel file
    df = pd.read_csv(filename)
    #or
    #df = pd.read_excel(filename)
    # parse the dates
    #was using the date as text but if you want it as date time, paul just change all places in the code
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    '''
    df.date_commande_client = pd.to_datetime(df['date_commande_client'],
                                             format='%m/%d/%y',
                                             errors='coerce')
    # rows with that could not be parsed are dropped
    df.dropna(axis=0, subset=['date_commande_client'], inplace=True)
    '''
    # Remove rows with no 'famille_intitule' ('prod_family')
    df.dropna(axis=0, subset=['prod_family'], inplace=True)
    # Remove rows from 'SERVICES'('prod_name')
    df.drop(df[df['prod_name'] ==  'SERVICES'].index, inplace=True)
    df['prod_id'] = df['prod_id'].astype(str)
    df['cont_id'] = df['cont_id'].astype(str)
    df['transaction_id'] = df['transaction_id'].astype(str)
    filter1 = df.groupby('cont_id')['transaction_id'].nunique().to_frame()
    filter1 = filter1[(filter1['transaction_id']>=6)]
    filter1=filter1.reset_index()
    allCus=list(set(list(filter1['cont_id'])))
    df = df[(df['cont_id'].isin(allCus))]

    filter2 = df.groupby('prod_name')['transaction_id'].nunique().to_frame()
    filter2 = filter2[(filter2['transaction_id']>=75)]
    filter2=filter2.reset_index()
    allProds=list(set(list(filter2['prod_name'])))
    df = df[(df['prod_name'].isin(allProds))]

    allProds = list(set(list(df['prod_name'])))
    invalidProds = []
    for p in allProds:
        rows = df[(df['prod_name']==p)]
        rows.sort_values(by=['transaction_date'])
        dates = list(rows['transaction_date'])
        lastDate = dates[-1]
        if lastDate<=datetime.datetime(2015,4,23):
            invalidProds.append(p)
    df = df[~(df['prod_name'].isin(invalidProds))]
    df = df.drop_duplicates(subset=None, keep='first', inplace=False)
    # Finer-grain categories: category1 and category2
    categories = pd.read_csv(cats_filename)
    category_by_famille = {(c, d): (a, b) for a, b, c, d in categories.values}
    df['category1'] = df.apply(
        lambda x: category_by_famille.get((x['prod_family'], x['prod_subfamily']), ('X', 'X'))[0],
        axis=1)
    df['category2'] = df.apply(
        lambda x: category_by_famille.get((x['prod_family'], x['prod_subfamily']), ('X', 'X'))[1],
        axis=1)
    # Remove rows with category1 as 'X'
    df.drop(df[df['category1'] ==  'X'].index, inplace=True)
    return df

def gen_model_ids(df):
    # some 'modele_intitule' are numbers (!)
    # make'em strings so that we can sort
    df['prod_name'] = df['prod_name'].astype(str)
    models = [k for k in df.fillna('').groupby(['prod_family', 'prod_subfamily', 'prod_name']).groups]
    models.sort()

    modelsdf = pd.DataFrame(
        {'model_id': i, 'prod_family': a, 'prod_subfamily': b, 'prod_name': c}
            for i, (a, b, c) in enumerate(models, start=1))
    #modelsdf.index = modelsdf.pop('model_id')
    return modelsdf

df = get_input_df(
    'myProject/static/file/clean.csv',
    'myProject/static/file/famillesall_04-25-2019.csv')
df['prod_name'] = df['prod_name'].astype(str)

models_df = gen_model_ids(df)

mergedf = df.merge(models_df, on=['prod_family', 'prod_subfamily', 'prod_name'])

# need the full name also
cat1s = {'L':'LIVING ROOM', 'D':'DINING ROOM', 'B':'BEDROOM', 'A':'ACCESSORIES',
'O': 'OTHER'}

# Mapping the dictionary keys to the data frame.
mergedf['CAT1'] = mergedf['category1'].map(cat1s)

cnx = sqlite3.connect('inbox_db.db')
mergedf.to_sql('inbox_table', con=cnx)
