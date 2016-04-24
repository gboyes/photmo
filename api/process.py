import os
import datetime
import json
import falcon
from model.material import Dictionary, Target
import model.analysis
from mapper import imagemapper

import gevent.monkey as monkey
monkey.patch_all()

import gevent

import time
OUTPUT_DIRECTORY = 'output'

class Process(object):

    def __init__(self):
        self.mapper = imagemapper.ImageMapper()
        self.output_path = OUTPUT_DIRECTORY

    def on_post(self, req, resp, image_id):
        body = req.stream.read()
        body = json.loads(body.decode('utf-8'))

        target_object = self.mapper.find(image_id)
        if target_object == None:
            resp.status = falcon.HTTP_404
            return

        target = Target(target_object['location'])

        dictionary_params = body['dictionary']
        dicitionary_limit = dictionary_params.get('max', 0)
        dictionary_images = [d.get('location') for d in self.mapper.find_many(dicitionary_limit)]
        dictionary = Dictionary(dictionary_images, dictionary_params)

        timestamp = datetime.datetime.now()
        filename = "{}_{}.png".format(target_object['image_id'], timestamp.strftime("%Y-%m-%d_%H_%M_%S"))

        output_location = os.path.join(self.output_path, filename)
        analysis = model.analysis.Analysis(target, dictionary, output_location, params=body['analysis'])

        gevent.spawn(analysis.start)
        resp.location = output_location
