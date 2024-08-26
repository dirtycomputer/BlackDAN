import json

def merge_json_files(file_list, output_file):
    merged_data = []

    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data)  # 合并到一个列表中

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

file_list = [f'multi_blackdan_results_gpu{i}.json' for i in range(1,5)]
output_file = 'merged_output.json'

merge_json_files(file_list, output_file)