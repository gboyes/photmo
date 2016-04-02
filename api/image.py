import falcon
import json
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
        resp.body = result

    def on_delete(self, req, resp, image_id):
        pass

class Images(object):
    def __init__(self):
        self.mapper = imagemapper.ImageMapper()
    def on_get(self, req, resp):
        resp.body = self.mapper.find_all()