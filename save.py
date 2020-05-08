'''
    def update_model_with_inputs(self):
        data = build(self.df_with_inputs)
        train_x,train_y = self.split_data(self.df_with_inputs,data)
        self.train_x_with_inputs = train_x
        #self.train_y_with_inputs = train_y
        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        model_knn.fit(self.train_x_with_inputs)
        # model with input is not a fixed one
        self.model_knn1 = model_knn
    def update_model1(self):
        df = self.loadData()
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

    def euclidean_distance(self,row1, row2):
        #return euclidean_distance between two data points
        distance = 0.0
        for i in range(len(row2)):
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

        def get_neighbors(self,train_x,train_y, test_row, num_neighbors):
            # Locate the most similar neighbors
            distances = []
            for i,train_row in enumerate(train_x):
                #print(len(test_row))
                #print(len(train_row))
                dist = self.euclidean_distance(test_row, train_row)

                distances.append((dist,train_y[i]))
            distances.sort(key=lambda tup: tup[0])
            neighbors = []
            for i in range(num_neighbors):
                neighbors.extend(distances[i][1])
            return neighbors

        def predict_classification(self,train_x,train_y, test_x, num_neighbors,num_recs):
            # Make a prediction based on nearest neighbors
            neighbors = self.get_neighbors(train_x,train_y, test_x, num_neighbors)
            output_values = {}
            for i in set(neighbors):
                if neighbors.count(i)>=1:
                    output_values[i] = neighbors.count(i)
            output_values = sorted(output_values.items(), key=lambda item:item[1],reverse=True)
            prediction = []
            for i in range(num_recs):
                if i >= len(output_values):
                    break
                prediction.append(output_values[i][0])
            #print("*************************")
            #print(prediction)
            return prediction

        def k_nearest_neighbors(self,train_x,train_y, test_x, num_neighbors,num_recs=6):
            # kNN Algorithm
            output = self.predict_classification(train_x,train_y,test_x, num_neighbors,num_recs)
            return(output)

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
    '''
