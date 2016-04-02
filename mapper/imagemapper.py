from bson.objectid import ObjectId
from bson.json_util import dumps
import app

class ImageMapper(object):
    def __init__(self):
        self.collection = app.db.photmo_images
    def insert(self, object):
        return self.collection.insert_one(object)
    def find_all(self):
        return dumps([self._flatten_id(i) for i in list(self.collection.find())])
    def find(self, object_id):
        return dumps(self._flatten_id(self.collection.find_one({'_id': ObjectId(object_id)})))

    def _flatten_id(self, object):
        ret = {}
        for k,v in object.iteritems():
            if k == '_id':
                ret['image_id'] = str(v)
                continue
            ret[k] = v
        return ret

