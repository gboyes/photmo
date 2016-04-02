import falcon

from api.analysis import Analysis

api = falcon.API()
api.add_route('/analysis', Analysis())
