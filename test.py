import os
from tqdm import tqdm
import json
import re
import shutil

data_root = "/data/dataset/MM-SafetyBench"
question_file_dir = os.path.join(data_root, 'processed_questions')
img_file_dir = os.path.join(data_root, 'img')
result_file_dir = "/home/haowang/xywang/BlackDAN/multimodal_results/test"

if os.path.exists(result_file_dir):
    shutil.rmtree(result_file_dir)

os.makedirs(result_file_dir)

question_files = os.listdir(question_file_dir)

type_list = ['SD', 'SD_TYPO', 'TYPO']
num = 0
pattern = re.compile(r'0[1-7]|09')
for question_file_name in tqdm(question_files):
    if not pattern.search(question_file_name):
        continue
    
    input_file_path = os.path.join(question_file_dir, question_file_name)
    image_file_path = os.path.join(img_file_dir, question_file_name[:-5])
    result = {}

    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for data_type in type_list:
        output_file_dir = os.path.join(result_file_dir, data_type)
        output_file_path = os.path.join(output_file_dir, question_file_name)
                                       
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)

        for data_id in tqdm(data):
            if data_type == 'SD_TYPO':
                question = data[data_id]['Rephrased Question']
            else:
                question = data[data_id]['Rephrased Question(SD)']

            image_query = os.path.join(image_file_path, data_type, f'{data_id}.jpg')

            result[data_id ] = {
                "result": f"{question}, {image_query}"
            }
            num+=1
        
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=4)
print(num)