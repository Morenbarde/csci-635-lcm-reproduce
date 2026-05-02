'''
    Functions for Causal Question and Statement Generation (Modules 2 & 3)
'''

import json
import os
import torch


def generate_causal_questions(model, domain, slice_name, output_folder="Output/"):

    topic_graph = [json.loads(line) for line in open(output_folder+"topic_graph_"+slice_name+".jsonl") if line.strip()]


    file_path = output_folder+"causal_questions_"+slice_name+".jsonl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as file:
        for entry in topic_graph:

            TOPIC = entry['topic']

            prompt =    f"You are an expert in {domain}. "\
                    f"Topic: '{TOPIC}'. "\
                    "Write 3 causal questions a student might ask about this topic. "\
                    "Each question should start with 'What causes' or 'What leads to'. "\
                    "Return only the questions, one per line.\n"\

            response = model.generate_response(prompt)
            question_list = response.strip().splitlines()

            entry["questions"] = question_list
            file.write(json.dumps(entry) + "\n")
            file.flush()


    torch.cuda.empty_cache()

    return



def generate_causal_statements(model, domain, slice_name, output_folder="Output/"):

    topic_graph = [json.loads(line) for line in open(output_folder+"topic_graph_"+slice_name+".jsonl") if line.strip()]


    file_path = output_folder+"causal_statements_"+slice_name+".jsonl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as file:
        for entry in topic_graph:

            TOPIC = entry['topic']

            prompt =    f"You are an expert in {domain}. "\
                    f"Topic: '{TOPIC}'. "\
                    "Write 3 short statements of the form 'X causes Y' or "\
                    "'X leads to Y' that describe causal relationships in this topic. "\
                    "Each statement should focus on a single mechanism. "\
                    "Return only the statements, one per line.\n"\

            response = model.generate_response(prompt)
            statement_list = response.strip().splitlines()

            entry["statements"] = statement_list
            file.write(json.dumps(entry) + "\n")
            file.flush()


    torch.cuda.empty_cache()

    return
