import numpy as np
import pandas as pd
import datetime


def filters(df,min_orders = 5):
    # only select customers whose # of orders > 4
    filter1 = df.groupby('cont_id')['transaction_id'].nunique().to_frame()
    filter1 = filter1[(filter1['transaction_id']>=min_orders)]
    filter1=filter1.reset_index()
    allCus=list(set(list(filter1['cont_id'])))
    #df = df[(df['cont_id'].isin(allCus))]
    allCus.sort()
    return allCus

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
        df=df.sort_values(by=['cont_id','transaction_date'])
        # get all customer ids
        # apply filter to select training dataset
        self.train_client_ids = filters(df)
        # add per-customer counts for each prod_name
        self.all_client_ids = list(set(list(df['cont_id'])))
        self.test_client_ids = list(set(self.all_client_ids).difference(set(self.train_client_ids)))
        print("*******************************")
        print(len(self.test_client_ids))
        self.all_client_ids.sort()
        self.df = df


    def build(self,df):
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
        df3 = df2[dummCols].groupby(df2['cont_id']).sum()
        x = np.array(df3).astype(np.float)
        x = self.normalize_dataset(x)

        data = pd.DataFrame(x,columns=dummCols)
        data['cont_id'] = self.all_client_ids
        self.data = data

    def Training_data(self,df):
        # get the x and y for training dataset
        pastDF = df.copy()
        pastDF = pastDF[(pastDF['cont_id'].isin(self.train_client_ids))]
        target = []
        for c in self.train_client_ids:
            row = df[(df['cont_id']==c)]
            # get all transaction id
            tranIDs = list(set(list(row['transaction_id'])))
            tranIDs.sort(reverse=False)
            first = tranIDs[0:int(len(tranIDs)*2/3)]
            second = tranIDs[int(len(tranIDs)*2/3):]
            secondRows = pastDF[(pastDF['transaction_id'].isin(second))]
            #get the prod_family of the latest purchase
            target.append(list(secondRows['prod_name']))
            # drop records on that day
            pastDF =  pastDF[~(pastDF['transaction_id'].isin(second))]
        self.train_y = target
        cateCols = {'prod_name'}
        dfCate = pd.DataFrame(pastDF,columns=cateCols)
        dfCate = pd.get_dummies(dfCate)
        dummCols = []
        for i in list(dfCate.columns):
            if i.find('_nan')==-1:
                dummCols.append(i)
        dfCate = pd.DataFrame(dfCate,columns=dummCols)
        df2 = pd.DataFrame(pastDF,columns=['cont_id'])
        df2 = df2.join(dfCate)
        df3 = df2[dummCols].groupby(df2['cont_id']).sum()
        x = np.array(df3).astype(np.float)
        x = self.normalize_dataset(x)
        self.train_x = x

    def euclidean_distance(self,row1, row2):
        #return euclidean_distance between two data points
        distance = 0.0
        for i in range(len(row1)):
            distance += (row1[i] - row2[i])*(row1[i] - row2[i])
        return sqrt(distance)
    def cosine_distance(self,row1,row2):
        #return cosine_distance between two data points
        distance = 0.0
        for i in range(len(row1)):
            distance += row1[i]*row2[i]
        mag1 = 0
        mag2 = 0
        for i in range(len(row1)):
            mag1 += row1[i]*row1[i]
        for i in range(len(row2)):
            mag2 += row2[i]*row2[i]
        distance = distance/(sqrt(mag1)*sqrt(mag2))
        return distance


    def random_client_id(self):
        # return a random client id
        return np.random.choice(self.all_client_ids)

    def get_history(self, client_id):
        # get records given a client id
        qry = db_session.query(purchases)
        df = pd.read_sql(qry.statement, qry.session.bind)
        # remove records whose prod_name contains 'spare' or 'service'
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df=df[df['prod_name'].str.lower().str.contains('spare')==False]
        df=df[df['prod_name'].str.lower().str.contains('service')==False]
        #convert data type
        df['cont_id'] = df['cont_id'].astype(str)
        df=df.sort_values(by=['cont_id','transaction_date'])
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

        self.df=self.df[['prod_family','prod_subfamily','prod_name','CAT1','category1','category2','model_id','cont_id','transaction_date','transaction_id']]

        lookup=self.df[['prod_family','prod_subfamily','prod_name','CAT1','category1','category2','model_id']]
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
        #check = temp.to_dict(orient='records')
        #print(check)
        temp=temp.values.tolist()
        #print("******************1")
        #print(temp)
        for i in range(len(model_id_lst)):
        	try:
        		temp[i].extend([client_id, today,'suggestion'])
        	except IndexError:
        		print ('Model ID does not exist in database')
        #print("******************2")
        #print(temp)
        temp = temp[:len(model_id_lst)]
        tempdf=pd.DataFrame(temp, columns = list(self.df))
        tempdf.drop_duplicates(inplace=True)
        self.tempdf=tempdf

        return self.tempdf


    def recommend1(self, client_id):
        dfCur = self.df.copy()
        self.build(dfCur)
        self.Training_data(dfCur)
        row1 = self.data[self.data['cont_id']==str(client_id)]
        x_test = np.array(row1).astype(np.float)
        x_test = x_test[0][:-1]

        x_train = self.train_x
        y_train = self.train_y
        #print("*********************************")
        #print(len(x_test))
        #print(len(x_train))
        #print(len(y_train))
        res = self.k_nearest_neighbors(x_train,y_train, x_test,10,10)
        recommendations = dfCur[dfCur['prod_name'].isin(res)]
        recommendations['sort_cat'] = pd.Categorical(recommendations['prod_name'], categories=res, ordered=True)
        recommendations.sort_values('sort_cat', inplace=True)
        recommendations.reset_index(inplace=True)
        return recommendations


    def recommend2(self, client_id):
        #dfTrain=self.dfTrain[['CAT1', 'model_id', 'famille_intitule','sous_famille_intitule', 'modele_intitule']].drop_duplicates()
        dfCur = self.df.copy()
        dfCur=dfCur[(dfCur['transaction_id']!='suggestion')]
        self.build(dfCur)
        self.Training_data(dfCur)
        row1 = self.data[self.data['cont_id']==str(client_id)]
        x_test = np.array(row1).astype(np.float)
        x_test = x_test[0][:-1]

        x_train = self.train_x
        y_train = self.train_y
        #print("*********************************")
        #print(len(x_test))
        #print(len(x_train))
        #print(len(y_train))
        res = self.k_nearest_neighbors(x_train,y_train, x_test,10,10)
        recommendations = dfCur[dfCur['prod_name'].isin(res)]
        recommendations['sort_cat'] = pd.Categorical(recommendations['prod_name'], categories=res, ordered=True)
        recommendations.sort_values('sort_cat', inplace=True)
        recommendations.reset_index(inplace=True)
        return recommendations

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


    def get_neighbors(self,train_x,train_y, test_row, num_neighbors):
        # Locate the most similar neighbors
        distances = []
        for i,train_row in enumerate(train_x):
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((dist,train_y[i]))
        distances.sort(key=lambda tup: tup[0])
        neighbors = []
        for i in range(num_neighbors):
            neighbors.extend(distances[i][1])
        return neighbors


    def predict_classification(self,train_x,train_y, test_x, num_neighbors,num_recs):
        # Make a prediction with neighbors
        neighbors = self.get_neighbors(train_x,train_y, test_x, num_neighbors)
        output_values = {}
        for i in neighbors:
            if neighbors.count(i)>=1:
                output_values[i] = neighbors.count(i)
        output_values = sorted(output_values.items(), key=lambda item:item[1],reverse=True)
        prediction = []
        for i in range(num_recs):
            if i >= len(output_values):
                break
            prediction.append(output_values[i][0])
        return prediction


    def k_nearest_neighbors(self,train_x,train_y, test_x, num_neighbors,num_recs=3):
        # kNN Algorithm
        output = self.predict_classification(train_x,train_y,test_x, num_neighbors,num_recs)
        return(output)
