import torch
from transformers import BertTokenizer, BertModel
from torch import _dynamo
import time

SUM = 0
NUM_REQS = 100

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to("cuda:0")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda:0")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
opt_model = torch._dynamo.optimize("inductor")(model)

for i in range(NUM_REQS/10):
    outputs = opt_model(**inputs)
for i in range(NUM_REQS):
    start = time.time()
    outputs = opt_model(**inputs)
    end = time.time()
    SUM = SUM + (end-start)
print("Elapsed time: ", SUM/NUM_REQS)

