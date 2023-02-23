from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs


save_dir = "./saved_models/dpr"

# Let's first get some files that we want to use
doc_dir = "doc_data"

# Convert files to dicts
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

document_store = InMemoryDocumentStore(similarity="dot_product")

# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(docs)

reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=document_store)
document_store.update_embeddings(reloaded_retriever)

# queries = ["How do I initialize weights in PyTorch?","How do I check if PyTorch is using the GPU?","How do I print the model summary in PyTorch?", "How do I save a trained model in PyTorch?","What does .contiguous() do in PyTorch?","What does model.eval() do in pytorch?", "What does model.train() do in PyTorch?", "What does .view() do in PyTorch?", "What's the difference between reshape and view in pytorch?", "Why do we need to call zero_grad() in PyTorch?"]
result = reloaded_retriever.retrieve(query="How do I initialize weights in PyTorch?",top_k=2)
print(result)