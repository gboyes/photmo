from model.material import Dictionary, Target
import model.analysis
import json
import falcon

class Analysis(object):
    def on_post(self, req, resp):
        body = req.stream.read()
        body = json.loads(body.decode('utf-8'))
        dictionary = Dictionary(body['dictionary'])
        target = Target(body['target']['image'])
        analysis = model.analysis.Analysis(target, dictionary, params=body['analysis'])
        analysis.start()

