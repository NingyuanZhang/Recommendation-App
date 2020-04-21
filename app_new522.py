from app import app
#import json
#from pprint import pprint, pformat
from flask import Flask, render_template, request, jsonify, session, redirect
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
#from flask import make_response
#from flask_wtf import Form
#from wtforms import StringField, SubmitField
from db_setup import init_db, db_session, engine
from models import top_by_cat2, purchases, dropdown_table_new
import sqlalchemy
from sqlalchemy.orm import sessionmaker

init_db()

def filter_by_order_count(df):
    #df=df[df['commande']!='suggestion']
    customer_num_orders = df.groupby(['cont_id', 'transaction_id']).size().reset_index().groupby(['cont_id']).size()

    # how many products each customer ordered
    customer_num_products = df.groupby(['cont_id', 'prod_id']).size().reset_index().groupby(['cont_id']).size()

    min_products, max_products = 0, 20
    min_orders, max_orders = 1, 4

    customers_minmax_products = customer_num_products[(customer_num_products >= min_products) &
                                                      (customer_num_products <= max_products)].index
    customers_minmax_orders = customer_num_orders[(customer_num_orders >= min_orders) &
                                                  (customer_num_orders <= max_orders)].index

    customers = set(customers_minmax_orders) & set(customers_minmax_products)

    return df[df['cont_id'].isin(customers)]




class Recommender(object):

    def __init__(self):
        qry = db_session.query(purchases)
        df = pd.read_sql(qry.statement, qry.session.bind)
        df=df[df['prod_name'].str.lower().str.contains('spare')==False]
        df=df[df['prod_name'].str.lower().str.contains('service')==False]

        df = filter_by_order_count(df)

        # add per-customer counts for category2 and model_id
        df['qty']=df.groupby(['cont_id'])['prod_name'].transform('size')
        ndf=df.groupby(['cont_id','prod_name'])['qty'].sum().reset_index()
        ndf['cont_id'] = ndf['cont_id'].astype(str)

        self.df = df
        self.ndf=ndf

        self.client_ids = list(np.sort(ndf.cont_id.unique()))

        items = ndf.pivot(index = 'prod_name', columns = 'cont_id', values = 'qty').fillna(0)
        self.items=items
        item_rows=csr_matrix(items.values)
        self.item_rows=item_rows

        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        model_knn.fit(item_rows)
        self.model_knn = model_knn


    def random_client_id(self):
        return np.random.choice(self.client_ids)

    def get_history(self, client_id):
        return self.df[self.df['cont_id'] == client_id]

    def get_model_ids(self, model_ids):
        columns = [
            'CAT1',
            'category1',
            'prod_family',

            'category2',
            'prod_subfamily',

            'prod_name',
            'model_id'
            #'code_article'
        ]

        fdf = self.df[self.df['model_id'].isin(model_ids)][columns].drop_duplicates()

        return fdf


    def update_history2(self, client_id, model_id1=None, model_id2=None, model_id3=None):

        self.df=self.df[['prod_family','prod_subfamily','prod_name','CAT1','category1','category2','model_id','cont_id','transaction_date','qty','transaction_id']]

        lookup=self.df[['prod_family','prod_subfamily','prod_name','CAT1','category1','category2','model_id',]]
        lookup.drop_duplicates(inplace=True)

        today=pd.to_datetime('today').strftime('%Y-%m-%d')

        model_id_lst=[]
        if model_id1:
            model_id_lst.append(model_id1)
        if model_id2:
            model_id_lst.append(model_id2)
        if model_id3:
            model_id_lst.append(model_id3)

        temp=lookup[lookup['prod_name'].isin(model_id_lst)]
        temp=temp.values.tolist()
        for i in range(len(model_id_lst)):
        	try:
        		temp[i].extend([client_id, today, 50, 'suggestion'])
        	except IndexError:
        		print ('Model ID does not exist in database')

        tempdf=pd.DataFrame(temp, columns = list(self.df))
        tempdf.drop_duplicates(inplace=True)
        self.tempdf=tempdf

        return self.tempdf


    def get_list(self):

    	grouped = self.df.groupby(['CAT1', 'model_id'])['model_id'].agg(
		    {"code_count": len}).sort_values("code_count", ascending=False).reset_index()
    	grouped = grouped.groupby('CAT1').head(5).reset_index().sort_values('CAT1', ascending=True)

    	return grouped


    def recommend(self, client_id):


        temp=self.ndf[self.ndf.cont_id==str(client_id)]
        self.temp=temp
        client_items=temp.sort_values('qty')['prod_name']
        self.client_items=client_items


        # find nearest neighbours for category2
        for val in client_items.values:
            distances, indices = self.model_knn.kneighbors(
                self.items.loc[str(val), :].values.reshape(1, -1), n_neighbors = 10)

        nearest_item_list = [
            self.items.index[indices.flatten()[i]]
            for i in range(0, len(distances.flatten()))
        ]

        main_list = np.setdiff1d(nearest_item_list,client_items.values)

        recommendations = self.df[self.df['prod_name'].isin(main_list)]
        recommendations['sort_cat'] = pd.Categorical(recommendations['prod_name'], categories=main_list, ordered=True)
        recommendations.sort_values('sort_cat', inplace=True)
        recommendations.reset_index(inplace=True)

        return recommendations


    def recommend2(self, client_id):
        #ndf=self.ndf[['CAT1', 'model_id', 'famille_intitule','sous_famille_intitule', 'modele_intitule']].drop_duplicates()
        qry = db_session.query(purchases)
        df = pd.read_sql(qry.statement, qry.session.bind)
        df=df[df['prod_name'].str.lower().str.contains('spare')==False]
        df=df[df['prod_name'].str.lower().str.contains('service')==False]

        df = filter_by_order_count(df)

        # add per-customer counts for category2 and model_id
        df['qty']=df.groupby(['cont_id'])['prod_name'].transform('size')
        ndf=df.groupby(['cont_id','prod_name'])['qty'].sum().reset_index()
        self.df = df
        self.ndf=ndf

        self.client_ids = list(np.sort(ndf.cont_id.unique()))

        items = ndf.pivot(index = 'prod_name', columns = 'cont_id', values = 'qty').fillna(0)
        self.items=items
        item_rows=csr_matrix(items.values)
        self.item_rows=item_rows

        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        model_knn.fit(item_rows)
        self.model_knn = model_knn

        temp=self.ndf[self.ndf.cont_id==(client_id)]
        self.temp=temp
        client_items=temp.sort_values('qty')['prod_name']
        self.client_items=client_items

        # find nearest neighbours for category2
        for val in client_items.values:
            distances, indices = self.model_knn.kneighbors(
                self.items.loc[str(val), :].values.reshape(1, -1), n_neighbors = 10)

        nearest_item_list = [
            self.items.index[indices.flatten()[i]]
            for i in range(0, len(distances.flatten()))
        ]

        main_list = np.setdiff1d(nearest_item_list,client_items.values)

        recommendations = self.df[self.df['prod_name'].isin(main_list)]
        recommendations['sort_cat'] = pd.Categorical(recommendations['prod_name'], categories=main_list, ordered=True)
        recommendations.sort_values('sort_cat', inplace=True)
        recommendations.reset_index(inplace=True)

        return recommendations



recommender = Recommender()


@app.route('/', methods=['GET','POST'])
def index():
	return render_template('index.html')

@app.route('/random_client', methods=["GET","POST"])
def random():
    qry = db_session.query(dropdown_table_new)
    df2 = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    for x in df2['prod_subfamily'].unique():
    	temp=df2[df2['prod_subfamily']==x]['prod_name'].unique()
    	list2.append((x, temp))
    # df2=df[['CAT1','modele_intitule']].drop_duplicates()
    # df3=df[['CAT1','sous_famille_intitule','modele_intitule']].drop_duplicates()
    df4=df2.sort_values(by=['prod_family'])
    df4=df4.groupby(['CAT1','prod_family'])['prod_subfamily'].unique().apply(list).reset_index()

    client_id = recommender.random_client_id()
    df_reco_models = recommender.recommend(client_id )
    #recommended_model_ids = recommendations['model_id'].unique()
    #df_reco_models = recommender.get_model_ids(recommended_model_ids)
    if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
        df_reco_models.sort_values('category1', ascending=False, inplace=True)
        df_reco_models.reset_index(inplace=True)
    else:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
        df_reco_models.sort_values('category1', ascending=False, inplace=True)
        df_reco_models.reset_index(inplace=True)
    print("*************id:"+str(client_id))
    df_history = recommender.get_history(client_id)
    columns= [
        ('transaction_date', 'ORDER DATE'),
        ('prod_family', 'FAMILY'),
        ('prod_subfamily', 'SUB-FAMILY'),
        ('prod_name', 'MODEL'),
        ('model_id', 'MODEL ID'),
        ('transaction_id', 'ORDER #')
    ]


    return render_template('reco80.html',
        client_id=client_id,
        history_df=df_history,
        history_columns=columns,
        #recommendations=pformat(recommendations),
        #recomm_df=df_reco_models.groupby('CAT1').head(1),
        #recomm_df=df_reco_models,
        recomm_df=df_reco_models.groupby('CAT1').head(1),
        df2=df2, list2=list2, df4=df4)



@app.route('/suggestion', methods=["GET","POST"])
def suggestion():
    qry = db_session.query(dropdown_table_new)
    df2 = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    for x in df2['prod_subfamily'].unique():
    	temp=df2[df2['prod_subfamily']==x]['prod_name'].unique()
    	list2.append((x, temp))
    # df2=df[['CAT1','modele_intitule']].drop_duplicates()
    # df3=df[['CAT1','sous_famille_intitule','modele_intitule']].drop_duplicates()
    df4=df2.sort_values(by=['prod_family'])
    df4=df4.groupby(['CAT1','prod_family'])['prod_subfamily'].unique().apply(list).reset_index()

    client_id=request.args['query']
    df_reco_models = recommender.recommend(client_id)
    # recommended_model_ids = recommendations['model_id'].unique()
    # df_reco_models=recommendations
    #df_reco_models = recommender.get_model_ids(recommended_model_ids)
    if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
    else:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
    df_history = recommender.get_history(client_id)
    columns= [
        ('transaction_date', 'ORDER DATE'),
        ('prod_family', 'FAMILY'),
        ('prod_subfamily', 'SUB-FAMILY'),
        ('prod_name', 'MODEL'),
        ('model_id', 'MODEL ID'),
        ('transaction_id', 'ORDER #')
    ]

    return render_template('reco10.html',
        client_id=client_id,
        history_df=df_history,
        history_columns=columns,
        #recommendations=pformat(recommendations),
        #recomm_df=df_reco_models,
        recomm_df=df_reco_models.groupby('CAT1').head(1),
        df2=df2, list2=list2, df4=df4
        )


@app.route('/db_add',methods=['GET','POST'])
def get_data():

    model_id1=request.form['suggestion1']
    model_id2=request.form['suggestion2']
    model_id3=request.form['suggestion3']
    client_id = request.form['client_id']

    df=recommender.update_history2(client_id, model_id1, model_id2, model_id3)
    listToWrite = df.to_dict(orient='records')

    Session = sessionmaker(bind=engine)
    session = Session()
    metadata = sqlalchemy.schema.MetaData(bind=engine,reflect=True)
    table = sqlalchemy.Table("inbox_table", metadata, autoload=True)

# Inser the dataframe into the database in one bulk
    conn = engine.connect()
    conn.execute(table.insert(), listToWrite)
# Commit the changes
    session.commit()
# Close the session
    session.close()
#get the other stuff to repopulate the page
    qry = db_session.query(dropdown_table_new)
    df2 = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    for x in df2['prod_subfamily'].unique():
    	temp=df2[df2['prod_subfamily']==x]['prod_name'].unique()
    	list2.append((x, temp))
    # df2=df[['CAT1','modele_intitule']].drop_duplicates()
    # df3=df[['CAT1','sous_famille_intitule','modele_intitule']].drop_duplicates()
    df4=df2.sort_values(by=['prod_family'])
    df4=df4.groupby(['CAT1','prod_family'])['prod_subfamily'].unique().apply(list).reset_index()

    client_id = recommender.random_client_id()
    df_reco_models = recommender.recommend2(client_id )
    #recommended_model_ids = recommendations['model_id'].unique()
    #df_reco_models = recommender.get_model_ids(recommended_model_ids)
    if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
        df_reco_models.sort_values('category1', ascending=False, inplace=True)
        df_reco_models.reset_index(inplace=True)
    else:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
        df_reco_models.sort_values('category1', ascending=False, inplace=True)
        df_reco_models.reset_index(inplace=True)

    df_history = recommender.get_history(client_id)
    columns= [
        ('transaction_date', 'ORDER DATE'),
        ('prod_family', 'FAMILY'),
        ('prod_subfamily', 'SUB-FAMILY'),
        ('prod_name', 'MODEL'),
        ('model_id', 'MODEL ID'),
        ('transaction_id', 'ORDER #')
    ]

    return render_template('reco80.html',
        client_id=client_id,
        history_df=df_history,
        history_columns=columns,
        #recommendations=pformat(recommendations),
        #recomm_df=df_reco_models.groupby('CAT1').head(1),
        #recomm_df=df_reco_models,
        recomm_df=df_reco_models.groupby('CAT1').head(1),
        df2=df2, list2=list2, df4=df4)



if __name__ == '__main__':
    app.run(debug=True)
