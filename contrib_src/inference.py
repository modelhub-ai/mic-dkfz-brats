import torch
import json
from processing import ImageProcessor
from modelhublib.model import ModelBase
from predict import segment


class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # load the DL models
        # load net
        self._model = 'model/params'

    def infer(self, input):
        output = segment(input["t1"]["fileurl"], input["t1c"]["fileurl"], input["t2"]["fileurl"],input["flair"]["fileurl"], self._model)
        output = self._imageProcessor.computeOutput(output)
        return output
