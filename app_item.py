from myProject import app
#import json
#from pprint import pprint, pformat
from flask import Flask, render_template, request, jsonify, session, redirect,url_for
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
#from flask import make_response
#from flask_wtf import Form
#from wtforms import StringField, SubmitField
from myProject.db_setup import init_db, db_session, engine
from myProject.models import top_by_cat2, purchases, dropdown_table_new
import sqlalchemy
from sqlalchemy.orm import sessionmaker
import os
import datetime
import time
from myProject.extraction import GetData


init_db()


class Recommender(object):

    def __init__(self):
        # get the inbox_table
        qry = db_session.query(purchases)
        df = pd.read_sql(qry.statement, qry.session.bind)
        # remove records whose prod_name contains 'spare' or 'service'
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df=df[df['prod_name'].str.lower().str.contains('spare')==False]
        df=df[df['prod_name'].str.lower().str.contains('service')==False]
        #convert data type
        df['cont_id'] = df['cont_id'].astype(str)
        df=df[(df['transaction_id']!='suggestion')]

        df=df.sort_values(by=['cont_id','transaction_date'])


        # get all customers
        self.client_ids = list(set(list(df['cont_id'])))
        # get customers in testing dataset

        self.client_ids = list(set(list(df['cont_id'])))
        df['qty']=df.groupby(['cont_id','prod_name'])['prod_name'].transform('size')
        self.df = df
        ndf=df.groupby(['cont_id','prod_name'])['qty'].sum().reset_index()
        self.ndf=ndf
        self.split_data()
        # get all candidates products
        items = ndf.pivot(index = 'prod_name', columns = 'cont_id', values = 'qty').fillna(0)
        self.items=items
        # compress sparse row marix
        item_rows=csr_matrix(items.values)
        self.item_rows=item_rows
        # build model
        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        # train model
        model_knn.fit(item_rows)
        self.model_knn2 = model_knn
        self.update_model1()

    def filters(self,df,min_orders = 4):
        # only select customers whose # of orders > 4
        filter1 = df.groupby('cont_id')['transaction_id'].nunique().to_frame()
        filter1 = filter1[(filter1['transaction_id']>=min_orders)]
        filter1=filter1.reset_index()
        allCus=list(set(list(filter1['cont_id'])))
        #df = df[(df['cont_id'].isin(allCus))]
        allCus.sort()
        return allCus

    def update_model1(self):
        qry = db_session.query(purchases)
        df = pd.read_sql(qry.statement, qry.session.bind)
        # remove records whose prod_name contains 'spare' or 'service'
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df=df[df['prod_name'].str.lower().str.contains('spare')==False]
        df=df[df['prod_name'].str.lower().str.contains('service')==False]
        #convert data type
        df['cont_id'] = df['cont_id'].astype(str)
        df=df.sort_values(by=['cont_id','transaction_date'])
        df['qty']=df.groupby(['cont_id','prod_name'])['prod_name'].transform('size')
        ndf=df.groupby(['cont_id','prod_name'])['qty'].sum().reset_index()
        # get all candidates products
        items = ndf.pivot(index = 'prod_name', columns = 'cont_id', values = 'qty').fillna(0)
        # compress sparse row marix
        item_rows=csr_matrix(items.values)
        # build model
        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        # train model
        model_knn.fit(item_rows)
        self.model_knn1 = model_knn

    def split_data(self):
        target = []
        history = []
        for c in self.client_ids:
            row = self.df[(self.df['cont_id']==c) & (self.df['transaction_id']!="suggestion")]
            # get all transaction id
            tranIDs = list(set(list(row['transaction_id'])))
            first = tranIDs[0:int(len(tranIDs)*2/3)]
            second = tranIDs[int(len(tranIDs)*2/3):]
            secondRows = row[(row['transaction_id'].isin(second))]
            #get the prod_name of the recent orders
            target.append(list(secondRows['prod_name']))
            # drop records on that day
            firstRows =  row[~(row['transaction_id'].isin(second))]
            history.append(list(firstRows['prod_name']))
        data = {"cont_id":self.client_ids,"history":history,"target":target}
        dfTarget = pd.DataFrame(data)
        self.dfTarget = dfTarget

    def random_client_id(self):
        # return a random client id
        return np.random.choice(self.client_ids)

    def get_history(self, client_id):
        # get records given a client id
        qry = db_session.query(purchases)
        df = pd.read_sql(qry.statement, qry.session.bind)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df=df[df['prod_name'].str.lower().str.contains('spare')==False]
        df=df[df['prod_name'].str.lower().str.contains('service')==False]
        #convert data type
        df['cont_id'] = df['cont_id'].astype(str)
        df=df.sort_values(by=['cont_id','transaction_date'])
        # update the df since users could insert a new record
        df['qty']=df.groupby(['cont_id','prod_name'])['prod_name'].transform('size')
        self.df = df

        return self.df[self.df['cont_id'] == client_id]

    def get_model_ids(self, model_ids):
        # get records given a model id
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

        df=self.df[['prod_family','prod_subfamily','prod_name','CAT1','category1','category2','model_id','cont_id','transaction_date','qty','transaction_id']]

        lookup=df[['prod_family','prod_subfamily','prod_name','CAT1','category1','category2','model_id']]
        lookup.drop_duplicates(inplace=True)

        today=pd.to_datetime('today')

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
            # "1" is the weight of new suggestion
        	try:
        		temp[i].extend([client_id, today, 1, 'suggestion'])
        	except IndexError:
        		print ('Model ID does not exist in database')
        temp = temp[:len(model_id_lst)]
        tempdf=pd.DataFrame(temp, columns = ['prod_family','prod_subfamily','prod_name','CAT1','category1','category2','model_id','cont_id','transaction_date','qty','transaction_id'])
        tempdf.drop_duplicates(inplace=True)
        self.tempdf=tempdf
        self.update_model1()

        return self.tempdf


    def recommend1(self, client_id):

        temp=self.ndf[self.ndf.cont_id==str(client_id)]
        self.temp=temp
        client_items=temp.sort_values('qty')['prod_name']
        #client_items = self.dfTarget[self.dfTarget["cont_id"]==client_id]["history"]
        self.client_items=client_items
        #self.history=history
        # find all products the customer bought

        # find nearest neighbours for category2
        for val in client_items.values:
            distances, indices = self.model_knn1.kneighbors(
                self.items.loc[str(val), :].values.reshape(1, -1), n_neighbors = 10)
        nearest_item_list = [
            self.items.index[indices.flatten()[i]]
            for i in range(0, len(distances.flatten()))
        ]
        invalid = list(self.dfTarget[self.dfTarget["cont_id"]==client_id]["history"])[0]
        main_list = np.setdiff1d(nearest_item_list,invalid)
        recommendations = self.df[self.df['prod_name'].isin(main_list)]
        list1 = list(recommendations['prod_name'])
        recommendations['sort_cat'] = pd.Categorical(list1, categories=main_list, ordered=True)
        recommendations = recommendations.sort_values('sort_cat')
        recommendations = recommendations.reset_index()
        return recommendations


    def recommend2(self, client_id):
        # get all candidates products
        #dfCur=dfCur[(dfCur['transaction_id']!='suggestion')]
        temp=self.ndf[self.ndf.cont_id==str(client_id)]
        self.temp=temp
        client_items=temp.sort_values('qty')['prod_name']
        #client_items = self.dfTarget[self.dfTarget["cont_id"]==client_id]["history"]
        self.client_items=client_items
        #self.history=history
        # find all products the customer bought

        # find nearest neighbours for category2
        for val in client_items.values:
            distances, indices = self.model_knn2.kneighbors(
                self.items.loc[str(val), :].values.reshape(1, -1), n_neighbors = 10)
        nearest_item_list = [
            self.items.index[indices.flatten()[i]]
            for i in range(0, len(distances.flatten()))
        ]
        invalid = list(self.dfTarget[self.dfTarget["cont_id"]==client_id]["history"])[0]
        main_list = np.setdiff1d(nearest_item_list,invalid)
        recommendations = self.df[self.df['prod_name'].isin(main_list)]
        list1 = list(recommendations['prod_name'])
        recommendations['sort_cat'] = pd.Categorical(list1, categories=main_list, ordered=True)
        recommendations = recommendations.sort_values('sort_cat')
        recommendations = recommendations.reset_index()
        return recommendations



recommender = Recommender()


@app.route('/', methods=['GET','POST'])
def index():
	return render_template('index.html')


@app.route('/random_client', methods=["GET","POST"])
def random():
    start = time.time()
    qry = db_session.query(dropdown_table_new)
    df2 = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    # get (prod_subfamily,[all unique prod_name])
    for x in df2['prod_subfamily'].unique():
    	temp=df2[df2['prod_subfamily']==x]['prod_name'].unique()
    	list2.append((x, temp))
    # df2=df[['CAT1','modele_intitule']].drop_duplicates()
    # df3=df[['CAT1','sous_famille_intitule','modele_intitule']].drop_duplicates()
    # get all unique prod_subfamily for each distinct CAT1 and prod_family
    df4=df2.sort_values(by=['prod_family'])
    df4=df4.groupby(['CAT1','prod_family'])['prod_subfamily'].unique().apply(list).reset_index()
    # get random customer id
    client_id = recommender.random_client_id()
    df_reco_models = recommender.recommend1(client_id )
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
    recomm_df=df_reco_models.groupby('CAT1').head(1)
    df_history = recommender.get_history(client_id)
    prod_list = []
    for row in recomm_df.itertuples():
        prod_list.append(getattr(row, 'prod_name'))
    columns= [
        ('transaction_date', 'ORDER DATE'),
        ('prod_family', 'FAMILY'),
        ('prod_subfamily', 'SUB-FAMILY'),
        ('prod_name', 'MODEL'),
        ('model_id', 'MODEL ID'),
        ('transaction_id', 'ORDER #')
    ]
    end = time.time()
    print("********************************running time: "+ str(end-start))
    return render_template('reco.html',
        client_id=client_id,
        history_df=df_history,
        history_columns=columns,
        #recommendations=pformat(recommendations),
        #recomm_df=df_reco_models.groupby('CAT1').head(1),
        #recomm_df=df_reco_models,
        recomm_df=recomm_df,
        df2=df2, list2=list2, df4=df4,prod=prod_list)



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
    df_reco_models = recommender.recommend1(client_id)
    # recommended_model_ids = recommendations['model_id'].unique()
    # df_reco_models=recommendations
    #df_reco_models = recommender.get_model_ids(recommended_model_ids)
    if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
    else:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
    df_history = recommender.get_history(client_id)
    prod_list = []
    recomm_df = df_reco_models.groupby('CAT1').head(1)
    for row in recomm_df.itertuples():
        prod_list.append(getattr(row, 'prod_name'))
    columns= [
        ('transaction_date', 'ORDER DATE'),
        ('prod_family', 'FAMILY'),
        ('prod_subfamily', 'SUB-FAMILY'),
        ('prod_name', 'MODEL'),
        ('model_id', 'MODEL ID'),
        ('transaction_id', 'ORDER #')
    ]

    return render_template('reco.html',
        client_id=client_id,
        history_df=df_history,
        history_columns=columns,
        #recommendations=pformat(recommendations),
        #recomm_df=df_reco_models,
        recomm_df=recomm_df,
        df2=df2, list2=list2, df4=df4,prod=prod_list
        )


@app.route('/db_add',methods=['GET','POST'])
def get_data():

    model_id1=request.form['suggestion1']
    model_id2=request.form['suggestion2']
    model_id3=request.form['suggestion3']
    client_id = request.form['client_id']

    df=recommender.update_history2(client_id, model_id1, model_id2, model_id3)
    listToWrite = df.to_dict(orient='records')
    print("***********************************")
    print(listToWrite)
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
    return redirect(url_for("index"))

@app.route('/metrics',methods=['GET','POST'])
def get_metrics():
    start = time.time()
    allPositive1 = 0
    allRec1 = 0
    truePositive1 = 0 # good recommendation
    falsePositive1 = 0
    allPositive2 = 0
    allRec2 = 0
    truePositive2 = 0 # good recommendation
    falsePositive2 = 0
    for c in recommender.client_ids:
        df_reco_models1 = recommender.recommend1(c)
        #df_reco_models2 = recommender.recommend2(c)
        if len(df_reco_models1[df_reco_models1['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
            df_reco_models1=df_reco_models1[df_reco_models1['category1'].isin(['L','D','B'])]
            df_reco_models1.sort_values('category1', ascending=False, inplace=True)
            df_reco_models1.reset_index(inplace=True)
        else:
            df_reco_models1=df_reco_models1[df_reco_models1['category1'].isin(['L','D','B', 'A'])]
            df_reco_models1.sort_values('category1', ascending=False, inplace=True)
            df_reco_models1.reset_index(inplace=True)
        '''
        if len(df_reco_models2[df_reco_models2['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
            df_reco_models2=df_reco_models2[df_reco_models2['category1'].isin(['L','D','B'])]
            df_reco_models2.sort_values('category1', ascending=False, inplace=True)
            df_reco_models2.reset_index(inplace=True)
        else:
            df_reco_models2=df_reco_models2[df_reco_models2['category1'].isin(['L','D','B', 'A'])]
            df_reco_models2.sort_values('category1', ascending=False, inplace=True)
            df_reco_models2.reset_index(inplace=True)
        '''
        recomm_df1=df_reco_models1.groupby('CAT1').head(3)
        #recomm_df2=df_reco_models2.groupby('CAT1').head(1)
        prod_list1 = []
        prod_list2 = []
        for row in recomm_df1.itertuples():
            prod_list1.append(getattr(row, 'prod_name'))
        '''
        for row in recomm_df2.itertuples():
            prod_list2.append(getattr(row, 'prod_name'))
        '''
        allRec1 += len(prod_list1)
        #allRec2 += len(prod_list2)
        df_history = recommender.dfTarget[recommender.dfTarget["cont_id"]==c]
        global allProd
        allProd = list(df_history["target"])[0]
        allPositive1 += len(allProd)
        #allPositive2 += len(allProd)

        for i in prod_list1:
            if i in allProd:
                truePositive1 += 1
        '''
        for i in prod_list2:
            if i in allProd:
                truePositive2 += 1
        '''
    end = time.time()
    print("********************************running time: "+ str(end-start))
    precision1 = truePositive1/allPositive1
    recall1 = truePositive1/allRec1
    #precision2 = truePositive2/allPositive2
    #recall2 = truePositive2/allRec2
    return render_template('metrics.html',precision1=precision1,recall1=recall1,precision2=0,recall2=0)
@app.route('/getAllReco',methods=['GET','POST'])
def get_reco():
    Reco1 = []
    Reco2 = []
    Reco3 = []
    Reco4 = []
    Reco5 = []
    for c in recommender.client_ids:
        df_reco_models = recommender.recommend1(c)
        if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
            df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
            df_reco_models.sort_values('category1', ascending=False, inplace=True)
            df_reco_models.reset_index(inplace=True)
        else:
            df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
            df_reco_models.sort_values('category1', ascending=False, inplace=True)
            df_reco_models.reset_index(inplace=True)
        recomm_df = df_reco_models.groupby('prod_name').head(5)

        curReco = list(recomm_df['prod_name'])
        if len(curReco)>0:
            Reco1.append(curReco[0])
        if len(curReco)>1:
            Reco2.append(curReco[1])
        if len(curReco)>2:
            Reco3.append(curReco[2])
        if len(curReco)>3:
            Reco4.append(curReco[3])
        if len(curReco)>4:
            Reco5.append(curReco[4])
    data = {"cont_id":recommender.client_ids,"prod_name1":Reco1,"prod_name2":Reco2,"prod_name3":Reco3,"prod_name4":Reco4,"prod_name5":Reco5}
    dfAllReco = pd.DataFrame(data)
    dfAllReco.to_csv("myProject/static/file/allReco.csv",index = False)
    return redirect(url_for('view_reco'))

@app.route('/viewAllReco',methods=['GET','POST'])
def view_reco():
    dfAllReco = pd.read_csv("myProject/static/file/allReco.csv")
    return render_template('viewAllReco.html',T = [dfAllReco.to_html(classes='mystyle',formatters={'Name': lambda x: '<b>' + x + '</b>'})]);

@app.route("/downloadAllReco",methods=['GET','POST'])
def downloadAllReco():
    dfAllReco = pd.read_csv("myProject/static/file/allReco.csv")
    return GetData(dfAllReco)()

@app.route("/reset",methods=['GET','POST'])
def reset():
    # remove the suggestions
    Session = sessionmaker(bind=engine)
    session = Session()
    query = [{'transaction_id': 'suggestion'}]
    conn = engine.connect()
    conn.execute("DELETE FROM inbox_table WHERE transaction_id = 'suggestion'")
    # Commit the changes
    session.commit()
    # Close the session
    session.close()
    return redirect(url_for('index'))

@app.route("/upload",methods=['GET','POST'])
def upload():
    if request.method == 'GET':
        return render_template('uploadForm.html')
    elif request.method == 'POST':
        file = request.files['file']
        num =  int(request.form['num'])
        print(num)
        # get the filename
        #filename = secure_filename(file.filename)
        # save it to static folder
        file.save("myProject/static/file/client.csv")
        Session = sessionmaker(bind=engine)
        session = Session()
        conn = engine.connect()
        try:
            conn.execute("DROP TABLE inbox_table")
        except sqlalchemy.exc.OperationalError:
            print("The table has been deleted")
        try:
            conn.execute("DROP TABLE dropdown_table_new")
        except sqlalchemy.exc.OperationalError:
            print("The table has been deleted")
        # Commit the changes
        session.commit()
        # Close the session
        session.close()
        # generate new tables
        g = generate("client.csv","famillesall_04-25-2019.csv",num)
        g.gen()
        return redirect(url_for('index'))

@app.route("/setting",methods=['GET','POST'])
def setting():
    return render_template('setting.html')

if __name__ == '__main__':
    app.run(debug=True)
