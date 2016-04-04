import falcon
import json
from bson.json_util import dumps
from mapper import imagemapper

class Image(object):
    def __init__(self):
        self.mapper = imagemapper.ImageMapper()

    def on_post(self, req, resp):
        body = req.stream.read()
        body = json.loads(body.decode('utf-8'))
        result = self.mapper.insert(body)

        if result != None:
            resp.status = falcon.HTTP_201
        else:
            resp.status = falcon.HTTP_400

    def on_get(self, req, resp, image_id):
        result = self.mapper.find(image_id)
        if result != None:
            resp.status = falcon.HTTP_200
            resp.body = dumps(result)
        else:
            resp.status = falcon.HTTP_404

    def on_delete(self, req, resp, image_id):
        result = self.mapper.delete(image_id)
        if result != None:
            resp.status = falcon.HTTP_200
        else:
            resp.status = falcon.HTTP_400

class Images(object):
    def __init__(self):
        self.mapper = imagemapper.ImageMapper()

    def on_get(self, req, resp):
        resp.body = dumps(self.mapper.find_many())