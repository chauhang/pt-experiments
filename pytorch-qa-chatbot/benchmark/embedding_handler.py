import logging

import torch
from InstructorEmbedding import INSTRUCTOR
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class EmbeddingModelHandler(BaseHandler):
    def __init__(self):
        super(EmbeddingModelHandler, self).__init__()
        self.initialized = False
        self.model = None
        self.device = "cuda:0"

    def initialize(self, ctx):
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        self.model = INSTRUCTOR(model_name)
        self.model.to(self.device)

    def preprocess(self, data):
        sentences = []
        for row in data:
            row = row["data"]
            row = row.decode("utf-8")
            sentences.append(row)

        return sentences

    def inference(self, data):

        embeddings = []
        for row in data:
            embedding = self.model.encode(row)
            print("Embedding: ", embedding.size)
            embedding = torch.from_numpy(embedding).to(self.device)
            embeddings.append(embedding)

        return torch.stack(embeddings)
