from .models import Research, Search, Tweet
from flask_restful import abort


def abort_if_research_doesnt_exist(research_id):
    research = Research.query.get(research_id)
    if not research:
        abort(404, message="Research doesn't exist")
    else:
        return research
