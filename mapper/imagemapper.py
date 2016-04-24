import os
import uuid
import photmo

class ImageMapper(object):
    def __init__(self):
        self.collection = photmo.db.photmo_images
        self.image_path = 'images'

    def insert(self, object, ext):
        image_id = str(uuid.uuid4())
        object['image_id'] = image_id
        object['location'] =  os.path.join(self.image_path, image_id + ext)
        result = self.collection.insert_one(object)
        if result == None:
            return result
        else:
            return self.find(object['image_id'])

    def find_many(self, limit=0):
        return [self._flatten_id(i) for i in list(self.collection.find().limit(limit))]

    def find(self, image_id):
        object = self.collection.find_one({'image_id': image_id})
        if object != None:
            return self._flatten_id(object)

    def delete(self, image_id):
        return self.collection.delete_one({'image_id': image_id})

    def _flatten_id(self, object):
        return {k:v for k, v in object.iteritems() if k != '_id'}

