import json
from vllm import LLM, SamplingParams

# 加载数据
prompts_name = 'multi_guard_sim_llama-2-7b-chat-hf'
with open(f'results/{prompts_name}.json', 'r') as f:
    data = json.load(f)

model_name = "vicuna-7b-v1.5"
target_model = LLM(model=f"/data/model/{model_name}", enforce_eager=True, trust_remote_code=True, gpu_memory_utilization=0.85)

new_data = []

# 收集所有的 pareto_prompts
all_pareto_prompts = []
for sample in data:
    harmful_question = sample['harmful_question']
    combined_fronts = sample['combined_fronts']
    for front in combined_fronts:
        for individual in front:
            if individual['rank'] == 0:
                all_pareto_prompts.append(harmful_question + " " + individual['prompt'])

# 一次性进行推理
pareto_outputs = target_model.generate(all_pareto_prompts, SamplingParams(max_tokens=100))

# 按顺序收集输出并存储
output_index = 0
for sample in data:
    id = sample['id']
    harmful_question = sample['harmful_question']
    combined_fronts = sample['combined_fronts']
    pareto_prompts = []
    pareto_responses = []
    
    current_pareto_outputs = []
    
    for front in combined_fronts:
        for individual in front:
            if individual['rank'] == 0:
                pareto_prompts.append(harmful_question + " " + individual['prompt'])
                current_pareto_outputs.append(pareto_outputs[output_index])
                pareto_responses.append(pareto_outputs[output_index].outputs[0].text)
                output_index += 1

    new_sample = {
        'id': id,
        'harmful_question': harmful_question,
        'pareto_outputs': [{'text': output.outputs[0].text} for output in current_pareto_outputs],
        'pareto_responses': pareto_responses
    }

    new_data.append(new_sample)

# 保存结果
output_filename = f'results/{prompts_name}_transfer_{model_name}.json'
with open(output_filename, 'w') as f:
    json.dump(new_data, f, indent=4)

print(f'Results saved to {output_filename}')
