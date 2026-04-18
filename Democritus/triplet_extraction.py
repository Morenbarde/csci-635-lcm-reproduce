'''
    Functions for Triplet Extraction (Module 4)
'''

import json
import os

from triplet_extract import extract



def extract_statement_triples(slice_name):

    file_path = "Causal_Triples/triples_"+slice_name+".jsonl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    causal_statements = [json.loads(line) for line in open("Causal_Statements/causal_statements_"+slice_name+".jsonl") if line.strip()]


    with open(file_path, "w") as file:
        for entry in causal_statements: # Per Topic

            statements = entry["statements"]
            for statement in statements: # Per Statement

                triplets = extract(statement)
                for t in triplets: # Per Triplet
                    triple_out = {"subject": t.subject, "relation": t.relation, "object": t.object}
                    file.write(json.dumps(triple_out) + "\n")

            file.flush()