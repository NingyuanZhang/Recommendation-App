from myProject import db

class top_by_cat2(db.Model):
    __tablename__ = "top_by_cat2"

    CAT1 = db.Column(db.String)
    index = db.Column(db.String, primary_key=True)
    prod_name = db.Column(db.String) # modele_intitule
    prod_family= db.Column(db.String) # famille_intitule
    prod_subfamily = db.Column(db.String) # sous_famille_intitule
    model_id = db.Column(db.String)

    def __repr__(self):
        return "{}".format(self.name)

class purchases(db.Model):
    __tablename__ = "inbox_table"

    cont_id = db.Column(db.String) # client
    transaction_id = db.Column(db.String, primary_key=True) # commande
    transaction_date = db.Column(db.String) # date_commande_client
    prod_id = db.Column(db.String) # code_article
    model_id = db.Column(db.String) # model_id
    prod_family = db.Column(db.String) # famille_intitule
    prod_subfamily = db.Column(db.String) # sous_famille_intitule
    prod_name = db.Column(db.String) # modele_intitule
    category1 = db.Column(db.String)
    category2 = db.Column(db.String)
    CAT1 = db.Column(db.String)
    #db.Integer

    def __repr__(self):
        return "{}".format(self.name)

class dropdown_table_new(db.Model):
    __tablename__ = "dropdown_table_new"

    CAT1 = db.Column(db.String)
    index =db.Column(db.String, primary_key=True)
    prod_name = db.Column(db.String) # modele_intitule
    prod_family = db.Column(db.String) # famille_intitule
    prod_subfamily = db.Column(db.String) # sous_famille_intitule


    def __repr__(self):
        return "{}".format(self.name)
