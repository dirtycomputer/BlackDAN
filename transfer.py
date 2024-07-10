import json
from vllm import LLM, SamplingParams

# 加载数据
prompts_name = 'question_response_llamaguard2'
with open(f'results/{prompts_name}.json', 'r') as f:
    data = json.load(f)

# 提取questions
questions = [sample['harmful_question'] + " " + sample['best_prompt'] for sample in data]

# 初始化模型
model_name = "vicuna-13b-1.5"
target_model = LLM(model=f"/data/model/{model_name}", enforce_eager=True, trust_remote_code=True, gpu_memory_utilization=0.85)

# 生成responses
outputs = target_model.generate(questions, SamplingParams(max_tokens=100))
responses = [item.outputs[0].text for item in outputs]

# 创建新数据
new_data = []
for i, sample in enumerate(data):
    new_sample = {
        'id': sample['id'],
        'harmful_question': sample['harmful_question'],
        'best_prompt': sample['best_prompt'],
        'response': responses[i]
    }
    new_data.append(new_sample)

# 保存结果
output_filename = f'{prompts_name}_transfer_{model_name}.json'
with open(output_filename, 'w') as f:
    json.dump(new_data, f, indent=4)

print(f'Results saved to {output_filename}')