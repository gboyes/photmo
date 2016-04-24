import falcon
import mimetypes
from bson.json_util import dumps
from mapper import imagemapper

class Image(object):
    def __init__(self):
        self.mapper = imagemapper.ImageMapper()

    def on_post(self, req, resp):
        ext = mimetypes.guess_extension(req.content_type)
        result = self.mapper.insert({}, ext)
        image_path = result['location']
        with open(image_path, 'wb') as image_file:
            while True:
                chunk = req.stream.read(4096)
                if not chunk:
                    break
                image_file.write(chunk)

        resp.status = falcon.HTTP_201
        resp.body = dumps(result)


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