import logging
from InstructorEmbedding import INSTRUCTOR
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    def __init__(self):
        super(ModelHandler, self).__init__()
        self.initialized = False
        self.model = None

    def initialize(self, ctx):
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        self.model = INSTRUCTOR(model_name)
        # Forcing to load on the first gpu
        self.model.to("cuda:0")

    def preprocess(self, data):
        row = data[0]["data"]
        row = row.decode("utf-8")
        return [row]

    def inference(self, data):
        return self.model.encode(data)
