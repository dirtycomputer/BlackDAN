import json
from pprint import pprint

import plotly.graph_objects as go
import plotly.express as px  # This helps to get a color palette
import math

def load_json_dict(file, index):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if item.get('id') == index:
                return item["combined_fronts"]
    return None

# 前20个sample
for id in range(20):
    sample = load_json_dict('/content/multi_guard_sim_llama-2-7b-chat-hf.json', id)

    fig = go.Figure()

    # Create a color palette with as many colors as the number of ranks
    colors = px.colors.qualitative.Plotly  # A default color palette from Plotly
    num_colors = len(colors)

    for rank, rank_list in enumerate(sample):
        # Sort the rank_list based on the 'all-MiniLM-L6-v2' values
        sorted_rank_list = sorted(rank_list, key=lambda item: item['fitnesses']['all-MiniLM-L6-v2'])
        
        # Extract sorted x and y values
        x = [item['fitnesses']['all-MiniLM-L6-v2'] for item in sorted_rank_list]
        y = [item['fitnesses']['llama_guard_2'] for item in sorted_rank_list]

        # 鼠标悬停时只显示前100个字符
        text = ["rank:"+str(item["rank"])+","+item["response"][:100] for item in sorted_rank_list]

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',  # Connect points with lines
            marker=dict(color=colors[rank % num_colors]),  # Assign a color based on rank
            line=dict(color=colors[rank % num_colors], width=2),  # Line color and width
            hovertext=text,  # Text to show when hovering
            hoverinfo='text',
            name=f'Rank {rank}'  # Optional: label for legend
        ))

    # Update layout
    fig.update_layout(
        title='Multi-Objective Optimization Results',
        xaxis_title='all-MiniLM-L6-v2',
        yaxis_title='llama_guard_2',
        hovermode='closest',
    )

    fig.show()
