
import os
import pickle
import json
import argparse

def load_and_save_docs(folder_path, source_path, source_name):

    docs_source = 'https://pytorch.org/docs/stable/'
    
    data = []
    
    with os.scandir(source_path) as entries:
        for entry in entries:
            if entry.is_file():
                with open(source_path + '/' + entry.name, "r", encoding='iso-8859-1') as f:
                    text = f.read()
                    page_url = docs_source + entry.name.split('.txt')[0] + '.html'
                    data.append({'text': text, 'metadata': {'source': page_url}})

            elif entry.is_dir():
                for i in os.listdir(source_path + '/' + entry.name):
                    if i.endswith('.txt'):
                        with open(source_path + '/' + entry.name + '/' + i, "r", encoding='iso-8859-1') as f:
                            text = f.read()
                            page_url = docs_source + entry.name + '/' + i.split('.txt')[0] + '.html'
                            metadata = {"source": page_url}
                            data.append({'text': text, 'metadata': {'source': page_url}})
                            
                    elif os.path.isdir(source_path + '/' + entry.name + '/' + i):
                        directory = source_path + '/' + entry.name + '/' + i
                        for file in os.listdir(directory):
                            with open(directory + '/' + file , "r", encoding='iso-8859-1') as f:
                                text = f.read()
                                page_url = docs_source + entry.name + '/' + i + '/' + file.split('.txt')[0] + '.html'
                                metadata = {"source": page_url}
                                data.append({'text': text, 'metadata': {'source': page_url}})
    
    output_folder = folder_path + '/' + source_name
    if not os.path.exists(output_folder):
        print(f'creating folder {output_folder}')
        os.makedirs(output_folder)
        
    print(f'saving data into {output_folder} as {source_name}.json')
    with open(f'{output_folder}/{source_name}.json', 'w') as f:
        json.dump(data, f)
        
    pickle.dump(data, open(f'{output_folder}/{source_name}.pkl', 'wb'))
    
#     return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and save data files.')
    parser.add_argument('--folder_path', type=str, default='knowledgebase', help='Path where the output files will be saved')
    parser.add_argument('--source_path', type=str, default='text', help='Path of the source directory where the text files are located')
    parser.add_argument('--source_name', type=str, default='file', help='Name of the output files')
    args = parser.parse_args()

    load_and_save_docs(args.folder_path, args.source_path, args.source_name)

