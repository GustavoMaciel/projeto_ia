from flask import Blueprint, request
from flask_login import current_user, login_user, logout_user, login_required
from sentimentalize import rest_api
from flask_restful import Resource

main = Blueprint('main', __name__)


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world!'}


rest_api.add_resource(HelloWorld, '/')
