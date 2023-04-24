## Steps for Data Preparation

Data Sources

1. [PyTorch Blogs](https://pytorch.org/blog/)
2. [PyTorch Docs](https://github.com/pytorch/pytorch/tree/main/docs)
3. [Stack Overflow Posts](https://stackoverflow.com/questions/tagged/pytorch)
4. [PyTorch Discuss Forum](https://discuss.pytorch.org/)


### PyTorch Blogs

Create a new folder named `knowledgebase`

```
mkdir knowledgebase
```

The script clones the pytorch blogs and extracts the blog contents.

Run the following command to prepare the json file

```
python extract_blogs.py --folder_path 'knowledgebase' --source_name 'blogs'
```

### PyTorch Docs

PyTorch docs are integrated with the Sphinx document scheme.

Follow the instructions from [Build the documentation](https://github.com/pytorch/pytorch#building-the-documentation)

Or

Clone the PyTorch repo and move into docs directory

```
git clone https://github.com/pytorch/pytorch.git

cd docs
```

Install the necessary requirements

```
pip install -r requirements.txt
```

Run the following command to generate text files from the docs

```
make text
```

Once the `text` directory is generated, run the following command 


```
python extract_docs.py.py --folder_path 'knowledgebase' --source_path text --source_name 'docs'
```

### Generate Token count

Run the following command to generate token counts

```
python token_count.py --file_path 'knowledgebase/docs/docs.json'

python token_count.py --file_path 'knowledgebase/blogs/blogs.json'
```

### Output structure

the dataset will be created in the following structure

```
$ tree knowledgebase/
knowledgebase/
├── blogs
│   ├── blogs.json
│   ├── blogs.pkl
│   └── token_counts.json
└── docs
    ├── docs.json
    ├── docs.pkl
    └── token_counts.json


```



