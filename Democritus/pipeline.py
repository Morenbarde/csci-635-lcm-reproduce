from .model import Democritus_Model
from .topic_graph import generate_topic_graph
from .causal_generation import generate_causal_questions, generate_causal_statements
from .triplet_extraction import extract_statement_triples
from .filter_triples import filter_file

import time
import sys


def run_full(domain, root_topics, topic_depth, slice_name, output_folder):
    
    # --- Load Model ---

    load_model_start = time.time()
    model = Democritus_Model()
    model_load_time = time.time() - load_model_start
    print("Model Load Time: ", model_load_time)
    print()
    sys.stdout.flush()


    # --- Generate Topic Graph (Module 1) ---

    generate_topics_start = time.time()
    generate_topic_graph(model, domain, root_topics, topic_depth, slice_name, output_folder)
    generate_topics_time = time.time() - generate_topics_start
    print("Topic Graph Generation Time: ", generate_topics_time)
    print()
    sys.stdout.flush()



    # --- Generate Causal Questions (Module 2) ---

    generate_questions_start = time.time()
    generate_causal_questions(model, domain, slice_name, output_folder)
    generate_questions_time = time.time() - generate_questions_start
    print("Causal Question Generation Time: ", generate_questions_time)
    print()
    sys.stdout.flush()


    # --- Generate Causal Statements (Module 3) ---

    generate_statements_start = time.time()
    generate_causal_statements(model, domain, slice_name, output_folder)
    generate_statements_time = time.time() - generate_statements_start
    print("Causal Statement Generation Time: ", generate_statements_time)
    print()
    sys.stdout.flush()




    # --- Extract Triplets (Module 4) ---
    triple_extraction_start = time.time()
    extract_statement_triples(slice_name, output_folder)
    triple_extraction_time = time.time() - triple_extraction_start
    print("Causal Triplets Extraction Time: ", triple_extraction_time)
    print()
    sys.stdout.flush()


    # --- Filter Triples (Custom Module 5) ---
    triple_filter_start = time.time()
    triple_file = "triples_"+slice_name+".jsonl"
    filter_file(triple_file, output_folder)
    triple_filter_time = time.time() - triple_filter_start
    print("Causal Triplets Filter Time: ", triple_filter_time)
    print()
    sys.stdout.flush()



