'''
    Functions for Topic Graph Generation (Module 1)
'''

import re
import json
import os
import torch

def parse_numbered_list(response: str) -> list[str]:
    subtopics = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading number/bullet markers: "1.", "1)", "1 -", "-", "•", "*"
        cleaned = re.sub(r'^(\d+[\.\)\-]\s*|[-•*]\s*)', '', line).strip()
        if cleaned:
            subtopics.append(cleaned)
    return subtopics



def generate_topic_graph_rec(model, domain, path, target_depth, current_depth, file_path):
    '''
        Recursively generates topics in a graph down to a desired depth
    '''

    if current_depth > target_depth:
        return

    prompt =    f"You are an expert in {domain}. "\
                f"Given the topic '{path[-1]}', list 10 important subtopics "\
                "that help explain its causes, consequences, or mechanisms. "\
                "Return ONLY a numbered list of subtopics, one per line, "\
                "with no explanations.\n"\

    response = model.generate_response(prompt)
    topic_list = parse_numbered_list(response)


    with open(file_path, "a") as file:

        for topic in topic_list:
            topic_entry = {"topic": topic, "path": path+[topic]}
            file.write(json.dumps(topic_entry) + "\n")

    for topic in topic_list:
        generate_topic_graph_rec(model, domain, path+[topic], target_depth, current_depth+1, file_path)

    torch.cuda.empty_cache()

    return




def generate_topic_graph(model, domain, topic_list, target_depth, slice_name=""):
    '''
        Takes and initial topic, and calls the recursive topic generation to build topic graph
    '''

    file_path = "Topic_Graphs/topic_graph_"+slice_name+".jsonl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write to JSON file
    with open(file_path, "w") as file:
        for topic in topic_list:
            topic_entry = {"topic": topic, "path": [topic], "depth": 0}
            file.write(json.dumps(topic_entry) + "\n")

    for topic in topic_list:
        generate_topic_graph_rec(model, domain, [topic], target_depth, 1, file_path)

    return