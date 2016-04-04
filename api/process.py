from model.material import Dictionary, Target
import model.analysis
import json
import falcon
from mapper import imagemapper

class Process(object):

    def __init__(self):
        self.mapper = imagemapper.ImageMapper()


    def on_post(self, req, resp, image_id):
        body = req.stream.read()
        body = json.loads(body.decode('utf-8'))

        target_object = self.mapper.find(image_id)
        if target_object == None:
            resp.status = falcon.HTTP_404
            return

        target = Target(target_object['location'])

        jd = body['dictionary']

        limit = jd['max'] or 0
        dictionary_images = [d.get('location') for d in self.mapper.find_many(limit)]

        dictionary = Dictionary(dictionary_images, jd)

        analysis = model.analysis.Analysis(target, dictionary, params=body['analysis'])
        analysis.start()


