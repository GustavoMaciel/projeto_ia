from sentimentalize import db, marshmallow
from marshmallow import post_load
from datetime import datetime


class Research(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(254), unique=False, nullable=False)
    description = db.Column(db.Text, nullable=False)
    date_created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    searches = db.relationship('Search', backref='research', lazy=True, cascade="all,delete-orphan")


class Search(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    query = db.Column(db.String(254), unique=False, nullable=False)
    date_created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    research_id = db.Column(db.Integer, db.ForeignKey('research.id'), nullable=False)
    tweets = db.relationship('Tweet', backref='search', lazy=True, cascade="all,delete-orphan")


class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    tweet = db.Column(db.String(290), unique=False, nullable=False)
    is_analyzed = db.Column(db.Boolean, nullable=False, default=False)
    analysis = db.Column(db.String(10), unique=False, nullable=False)
    search_id = db.Column(db.Integer, db.ForeignKey('search.id'), nullable=False)


class ResearchSchema(marshmallow.ModelSchema):
    class Meta:
        model = Research


class SearchSchema(marshmallow.ModelSchema):
    class Meta:
        model = Search


class TweetSchema(marshmallow.ModelSchema):
    class Meta:
        model = Tweet

