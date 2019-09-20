from flask import Blueprint, request
from sentimentalize import rest_api, db
from flask_restful import Resource

from .utils import abort_if_research_doesnt_exist
from .models import Research, ResearchSchema, Search, SearchSchema, Tweet, TweetSchema

research = Blueprint('research', __name__)


class ResearchController(Resource):

    def get(self, research_id):
        _research = abort_if_research_doesnt_exist(research_id)
        return ResearchSchema().dumps(_research), 201

    def put(self, research_id):
        _research = abort_if_research_doesnt_exist(research_id)
        json_data = request.get_json()
        _research.name = json_data['name']
        _research.description = json_data['description']

        db.session.commit()

        return ResearchSchema().dumps(_research), 201

    def delete(self, research_id):
        _research = abort_if_research_doesnt_exist(research_id)
        db.session.delete(_research)
        db.session.commit()
        return {'message': 'Research successfully deleted', 'status': 201}, 201


class ResearchListController(Resource):

    def post(self):
        pass


rest_api.add_resource(ResearchController, '/research/<id>')
rest_api.add_resource(ResearchListController, '/research')
