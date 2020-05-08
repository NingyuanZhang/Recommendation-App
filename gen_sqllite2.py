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
    # rows with that could not be parsed are dropped
    df.dropna(axis=0, subset=['date_commande_client'], inplace=True)
    '''
    # Remove rows with no 'famille_intitule' ('prod_family')
    df.dropna(axis=0, subset=['prod_family'], inplace=True)
    df.dropna(axis=0, subset=['prod_subfamily'], inplace=True)
    df.dropna(axis=0, subset=['prod_name'], inplace=True)
    # Remove rows from 'SERVICES'('prod_name')
    df.drop(df[df['prod_name'] ==  'SERVICES'].index, inplace=True)
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

    allprods = list(set(list(df['prod_name'])))
    allfam = []
    allsubfam = []
    for p in allprods:
        row =  df[(df['prod_name']==p)]
        allfam.append(list(row['prod_family'])[0])
        allsubfam.append(list(row['prod_subfamily'])[0])
    c = {'prod_name':allprods,'prod_family':allfam,'prod_subfamily':allsubfam}
    dfNew = pd.DataFrame(c)
    # Finer-grain categories: category1 and category2
    categories = pd.read_csv(cats_filename)
    category_by_famille = {(c, d): (a, b) for a, b, c, d in categories.values}
    dfNew['category1'] = dfNew.apply(
        lambda x: category_by_famille.get((x['prod_family'], x['prod_subfamily']), ('X', 'X'))[0],
        axis=1)
    # Remove rows with category1 as 'X'
    dfNew.drop(dfNew[dfNew['category1'] ==  'X'].index, inplace=True)
    cat1s = {'L':'LIVING ROOM', 'D':'DINING ROOM', 'B':'BEDROOM', 'A':'ACCESSORIES',
    'O': 'OTHER'}
    # Mapping the dictionary keys to the data frame.
    dfNew['CAT1'] = dfNew['category1'].map(cat1s)
    order = ['CAT1', 'prod_family', 'prod_subfamily', 'prod_name',]
    dfNew = dfNew[order]
    return dfNew



dfNew = get_input_df(
    'myProject/static/file/clean.csv',
    'myProject/static/file/famillesall_04-25-2019.csv')
dfNew['prod_name'] = dfNew['prod_name'].astype(str)
cnx = sqlite3.connect('inbox_db.db')
dfNew.to_sql('dropdown_table_new', con=cnx)
