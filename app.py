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
from myProject.gen_sqllite import generate
import sqlalchemy
from sqlalchemy.orm import sessionmaker
import os
import datetime
import time
from myProject.extraction import GetData
from math import sqrt
from sklearn.neighbors import NearestNeighbors


init_db()



class Recommender(object):
    def __init__(self):
        # get the inbox_table
        df = self.loadData()
        # apply filter to select customers in training dataset
        self.train_client_ids = self.filters(df,1)
        # get all customers
        self.all_client_ids = list(set(list(df['cont_id'])))
        # get customers in testing dataset
        self.test_client_ids = list(set(self.all_client_ids).difference(set(self.train_client_ids)))
        self.all_client_ids.sort()
        self.test_client_ids.sort()
        self.train_client_ids.sort()
        self.model_with_inputs()
        self.model_without_inputs()

    def loadData(self):
        qry = db_session.query(purchases)
        df = pd.read_sql(qry.statement, qry.session.bind)
        # remove records whose prod_name contains 'spare' or 'service'
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df=df[df['prod_name'].str.lower().str.contains('spare')==False]
        df=df[df['prod_name'].str.lower().str.contains('service')==False]
        #convert data type
        df['cont_id'] = df['cont_id'].astype(str)
        #sort by customer_id and transaction date
        df=df.sort_values(by=['cont_id','transaction_date'])
        self.df_with_inputs = df
        df_without_inputs = df[(df['transaction_id']!='suggestion')]
        self.df_without_inputs = df_without_inputs
        return df_without_inputs

    def filters(self,df,min_orders = 4):
        # only select customers whose # of orders > 4
        filter1 = df.groupby('cont_id')['transaction_id'].nunique().to_frame()
        filter1 = filter1[(filter1['transaction_id']>=min_orders)]
        filter1=filter1.reset_index()
        allCus=list(set(list(filter1['cont_id'])))
        #df = df[(df['cont_id'].isin(allCus))]
        allCus.sort()
        return allCus

    def build(self,df):
        # build a new dataframe which is suitable fot KNN model
        # one-hot encoding for categorical columns
        cateCols = {'prod_name'}
        dfCate = pd.DataFrame(df,columns=cateCols)
        dfCate = pd.get_dummies(dfCate)
        dummCols = []
        for i in list(dfCate.columns):
            if i.find('_nan')==-1:
                dummCols.append(i)
        dfCate = pd.DataFrame(dfCate,columns=dummCols)
        df2 = pd.DataFrame(df,columns=['cont_id'])
        df2 = df2.join(dfCate)
        # group records by customer id
        df3 = df2[dummCols].groupby(df2['cont_id']).sum()
        x = np.array(df3).astype(np.float)
        # rescale dataset columns to the range 0-1
        x = self.normalize_dataset(x)
        data = pd.DataFrame(x,columns=dummCols)
        data['cont_id'] = self.all_client_ids
        return data

    def split_data(self,df,data):
        # split data and get the x and y for training dataset
        target = []
        for c in self.train_client_ids:
            row = df[(df['cont_id']==c)]
            # get all transaction id
            tranIDs = list(set(list(row['transaction_id'])))
            first = tranIDs[0:int(len(tranIDs)*2/3)]
            second = tranIDs[int(len(tranIDs)*2/3):]
            secondRows = df[(df['transaction_id'].isin(second))]
            #get the prod_name of the recent orders
            target.append(list(secondRows['prod_name']))
        #self.train_y = target
        dfX = data[(data['cont_id'].isin(self.train_client_ids))]
        dfX = dfX.drop(['cont_id'], axis=1)
        x = np.array(dfX).astype(np.float)
        #self.train_x = x
        return x,target

    def random_client_id(self):
        # return a random client id
        return np.random.choice(self.all_client_ids)

    def get_history(self, client_id):
        # get records given a client id
        df = self.loadData()
        # update the df since users could insert a new record
        return self.df_with_inputs[self.df_with_inputs['cont_id'] == client_id]

    def dataset_minmax(self,dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax

    def normalize_dataset(self,dataset):
        # Rescale dataset columns to the range 0-1
        minmax = self.dataset_minmax(dataset)
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        return dataset

    def model_with_inputs(self):
        data = self.build(self.df_with_inputs)
        self.data_with_inputs = data
        train_x,train_y = self.split_data(self.df_with_inputs,data)
        self.train_x_with_inputs = train_x
        self.train_y_with_inputs = train_y
        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        model_knn.fit(self.train_x_with_inputs)
        # model with input is not a fixed one
        self.model_knn1 = model_knn

    def model_without_inputs(self):
        data = self.build(self.df_without_inputs)
        self.data_without_inputs = data
        train_x,train_y = self.split_data(self.df_without_inputs,data)
        self.train_x_without_inputs = train_x
        self.train_y_without_inputs = train_y
        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        model_knn.fit(self.train_x_without_inputs)
        # model without input is a fixed one
        self.model_knn2 = model_knn

    def update_history(self, client_id, model_id1=None, model_id2=None, model_id3=None):
        # get the dataframe about new suggestions
        df=self.df_with_inputs[['prod_family','prod_subfamily','prod_name','CAT1','category1','category2','model_id','cont_id','transaction_date','transaction_id']]
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
        	try:
        		temp[i].extend([client_id, today,'suggestion'])
        	except IndexError:
        		print ('Model ID does not exist in database')
        temp = temp[:len(model_id_lst)]
        tempdf=pd.DataFrame(temp, columns = ['prod_family','prod_subfamily','prod_name','CAT1','category1','category2','model_id','cont_id','transaction_date','transaction_id'])
        tempdf.drop_duplicates(inplace=True)
        self.tempdf=tempdf
        return self.tempdf

    def recommend1(self, client_id,num_recs=5):
        # recommend with inputs
        dfCur = self.df_with_inputs.copy()
        row = self.data_with_inputs[self.data_with_inputs['cont_id']==str(client_id)]
        x_test = np.array(row).astype(np.float)
        x_test = x_test[0][:-1]
        distances, indices = self.model_knn1.kneighbors([x_test],10)
        res = {}
        for i in indices[0]:
            candidates = self.train_y_with_inputs[i]
            for p in set(candidates):
                res[p] = candidates.count(p)
        res = sorted(res.items(), key=lambda item:item[1],reverse=True)
        prediction = []
        for i in range(num_recs):
            if i >= len(res):
                break
            prediction.append(res[i][0])
        recommendations = dfCur[dfCur['prod_name'].isin(prediction)]
        recommendations['sort_cat'] = pd.Categorical(recommendations['prod_name'], categories=prediction, ordered=True)
        recommendations.sort_values('sort_cat', inplace=True)
        recommendations.reset_index(inplace=True)
        return recommendations


    def recommend2(self, client_id,num_recs=5):
        dfCur = self.df_without_inputs.copy()
        row = self.data_without_inputs[self.data_without_inputs['cont_id']==str(client_id)]
        x_test = np.array(row).astype(np.float)
        x_test = x_test[0][:-1]
        '''
        self.build(dfCur)
        self.split_data(dfCur)

        res = self.model_knn1(selx_train,y_train, x_test,10,10)
        '''
        distances, indices = self.model_knn1.kneighbors([x_test],10)
        res = {}
        for i in indices[0]:
            candidates = self.train_y_without_inputs[i]
            for p in set(candidates):
                res[p] = candidates.count(p)
        res = sorted(res.items(), key=lambda item:item[1],reverse=True)
        prediction = []
        for i in range(num_recs):
            if i >= len(res):
                break
            prediction.append(res[i][0])
        recommendations = dfCur[dfCur['prod_name'].isin(prediction)]
        recommendations['sort_cat'] = pd.Categorical(recommendations['prod_name'], categories=prediction, ordered=True)
        recommendations.sort_values('sort_cat', inplace=True)
        recommendations.reset_index(inplace=True)
        return recommendations


recommender = Recommender()


@app.route('/', methods=['GET','POST'])
def index():
	return render_template('index.html')


@app.route('/random_client', methods=["GET","POST"])
def random():
    print(len(recommender.train_client_ids))
    start = time.time()
    qry = db_session.query(dropdown_table_new)
    df2 = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    # get (prod_subfamily,[all unique prod_name])
    for x in df2['prod_subfamily'].unique():
    	temp=df2[df2['prod_subfamily']==x]['prod_name'].unique()
    	list2.append((x, temp))
    # get all unique prod_subfamily for each distinct CAT1 and prod_family
    df4=df2.sort_values(by=['prod_family'])
    df4=df4.groupby(['CAT1','prod_family'])['prod_subfamily'].unique().apply(list).reset_index()
    # get random customer id
    client_id = recommender.random_client_id()
    df_reco_models = recommender.recommend1(client_id,3)
    if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
        df_reco_models.sort_values('category1', ascending=False, inplace=True)
        df_reco_models.reset_index(inplace=True)
    else:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
        df_reco_models.sort_values('category1', ascending=False, inplace=True)
        df_reco_models.reset_index(inplace=True)
    recomm_df=df_reco_models.groupby('CAT1').head(1)
    # get recommendation list
    prod_list = []
    for row in recomm_df.itertuples():
        prod_list.append(getattr(row, 'prod_name'))
    #get purchasing history
    df_history = recommender.get_history(client_id)
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
        recomm_df=recomm_df,
        df2=df2, list2=list2, df4=df4,prod=prod_list)

@app.route('/suggestion', methods=["GET","POST"])
def suggestion():
    start = time.time()
    qry = db_session.query(dropdown_table_new)
    df2 = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    for x in df2['prod_subfamily'].unique():
    	temp=df2[df2['prod_subfamily']==x]['prod_name'].unique()
    	list2.append((x, temp))
    df4=df2.sort_values(by=['prod_family'])
    df4=df4.groupby(['CAT1','prod_family'])['prod_subfamily'].unique().apply(list).reset_index()
    client_id=request.args['query']
    df_reco_models = recommender.recommend1(client_id)
    if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
    else:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
    #get purchasing history
    df_history = recommender.get_history(client_id)
    recomm_df=df_reco_models.groupby('CAT1').head(1)
    # get recommendation list
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
        #recomm_df=df_reco_models,
        recomm_df=df_reco_models.groupby('CAT1').head(1),
        df2=df2, list2=list2, df4=df4,prod=prod_list
        )


@app.route('/db_add',methods=['GET','POST'])
def get_data():
    # insert inputs to table
    model_id1=request.form['suggestion1']
    model_id2=request.form['suggestion2']
    model_id3=request.form['suggestion3']
    client_id = request.form['client_id']
    df=recommender.update_history(client_id, model_id1, model_id2, model_id3)
    listToWrite = df.to_dict(orient='records')
    #print("***********************************")
    #print(listToWrite)
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
    return redirect(url_for("index"))

@app.route('/metrics',methods=['GET','POST'])
def get_metrics():
    # get performance metrics
    start = time.time()
    allPositive1 = 0
    allRec1 = 0
    truePositive1 = 0 # good recommendation
    falsePositive1 = 0
    allPositive2 = 0
    allRec2 = 0
    truePositive2 = 0 # good recommendation
    falsePositive2 = 0
    for c in recommender.train_client_ids:
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
        #prod_list2 = []
        for row in recomm_df1.itertuples():
            prod_list1.append(getattr(row, 'prod_name'))
        '''
        for row in recomm_df2.itertuples():
            prod_list2.append(getattr(row, 'prod_name'))
        '''
        allRec1 += len(prod_list1)
        #allRec2 += len(prod_list2)
        df_history = recommender.get_history(c)
        allProd = set(list(df_history['prod_name']))
        allPositive1 += len(allProd)
        #allPositive2 += len(allProd)
        for i in set(prod_list1):
            if i in set(allProd):
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
    # get recommendation for every customer
    start = time.time()
    Reco1 = []
    Reco2 = []
    Reco3 = []
    Reco4 = []
    Reco5 = []
    for c in recommender.test_client_ids:
        df_reco_models = recommender.recommend1(c,8)
        if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
            df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
            df_reco_models.sort_values('category1', ascending=False, inplace=True)
            df_reco_models.reset_index(inplace=True)
        else:
            df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
            df_reco_models.sort_values('category1', ascending=False, inplace=True)
            df_reco_models.reset_index(inplace=True)
        recomm_df = df_reco_models.groupby('prod_name').head(1)

        curReco = list(recomm_df['prod_name'])
        if len(curReco)>0:
            Reco1.append(curReco[0])
        else:
            Reco1.append(" ")
        if len(curReco)>1:
            Reco2.append(curReco[1])
        else:
            Reco2.append(" ")
        if len(curReco)>2:
            Reco3.append(curReco[2])
        else:
            Reco3.append(" ")
        if len(curReco)>3:
            Reco4.append(curReco[3])
        else:
            Reco4.append(" ")
        if len(curReco)>4:
            Reco5.append(curReco[4])
        else:
            Reco5.append(" ")
    end = time.time()
    print("********************************running time: "+ str(end-start))
    data = {"cont_id":recommender.test_client_ids,"recommendation_1":Reco1,"recommendation_2":Reco2,"recommendation_3":Reco3,"recommendation_4":Reco4,"recommendation_5":Reco5}
    dfAllReco = pd.DataFrame(data)
    dfAllReco.to_csv("myProject/static/file/allReco.csv",index = False)
    return redirect(url_for('view_reco'))

@app.route('/viewAllReco',methods=['GET','POST'])
def view_reco():
    # view the recommendation list for every customer
    dfAllReco = pd.read_csv("myProject/static/file/allReco.csv")
    return render_template('viewAllReco.html',T = [dfAllReco.to_html(classes='mystyle',formatters={'Name': lambda x: '<b>' + x + '</b>'})]);

@app.route("/downloadAllReco",methods=['GET','POST'])
def downloadAllReco():
    # download the recommendation
    dfAllReco = pd.read_csv("myProject/static/file/allReco.csv")
    return GetData(dfAllReco)()

@app.route("/reset",methods=['GET','POST'])
def reset():
    # remove the suggestions
    Session = sessionmaker(bind=engine)
    session = Session()
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
