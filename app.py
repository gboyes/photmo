import falcon
from pymongo import MongoClient

from api.process import Process
from api.image import Image, Images

db = MongoClient().photmo

api = falcon.API()
api.add_route('/process', Process())
api.add_route('/process/{image_id}', Process())
api.add_route('/image', Image())
api.add_route('/image/{image_id}', Image())
api.add_route('/images', Images())
