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

Follow the instructions from [Build the documentation](https://github.com/pytorch/pytorch#building-the-documentation).

Once the text files are generated, zip the folder as `text.zip`

Or

Unzip the [text.zip](text.zip) file for already built docs

Once the `text` archive is extracted, run the following command 


```
python extract_docs.py --folder_path 'knowledgebase' --source_path text --source_name 'docs'
```

### PyTorch Discuss forums

The following script extracts the links and posts of the top solved posts from PyTorch Discussion forum. The top solved posts are available in `discuss_solved_id.csv` file.

```
python extract_forums.py --folder_path 'knowledgebase' --source_name 'discussion_forum'
```

The top posts are extracted and saved inside knowledgebase folder.


### Stack Overflow Posts

The StackOverflow Posts archive is 18GB is size and when extracted, it generates a file - posts.xml which is 92GB in size.

To process the file large cpu memory is needed. 

PyTorch question and answers are saved in the format of CSV file using the below notebook.
[SO-Parser.ipynb](SO-Parser.ipynb)

or

Use the pre-created csv file `pt_question_answers.csv`

```
python extract_so_posts_from_csv.py --folder_path 'knowledgebase' --source_name 'stack_overflow'
```


### Generate Token count

Run the following command to generate token counts

```bash
python token_count.py --file_path 'knowledgebase/docs/docs.json'

python token_count.py --file_path 'knowledgebase/blogs/blogs.json'

python token_count.py --file_path 'knowledgebase/forums/discussion_forum.json'

python token_count.py --file_path 'knowledgebase/stack_overflow/stack_overflow.json'

```

### Output structure

the dataset will be created in the following structure

```
$ tree
.
├── blogs
│   ├── blogs.json
│   ├── blogs.pkl
│   └── token_counts.json
├── docs
│   ├── docs.json
│   ├── docs.pkl
│   └── token_counts.json
├── forums
│   ├── discussion_forum.json
│   ├── discussion_forum.pkl
│   └── token_counts.json
└── stack_overflow
    ├── stack_overflow.json
    ├── stack_overflow.pkl
    └── token_counts.json

```



