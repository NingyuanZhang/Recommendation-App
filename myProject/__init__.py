# app.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import timedelta

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///inbox_db.db'
app.secret_key = "flask rocks!"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

db = SQLAlchemy(app)
