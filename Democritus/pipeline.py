from model import Democritus_Model
from topic_graph import generate_topic_graph
from causal_generation import generate_causal_questions, generate_causal_statements
from triplet_extraction import extract_statement_triples

import time


if __name__ == "__main__":

    domain = "macroeconomics and financial markets"
    TOPIC = "Economics"
    slice_name = "econ" # For File Naming


    # --- Load Model ---

    load_model_start = time.time()
    model = Democritus_Model()
    model_load_time = time.time() - load_model_start
    print("Model Load Time: ", model_load_time)
    print()


    # --- Generate Topic Graph (Module 1) ---

    generate_topics_start = time.time()
    generate_topic_graph(model, domain, TOPIC, 0, slice_name)
    generate_topics_time = time.time() - generate_topics_start
    print("Topic Graph Generation Time: ", generate_topics_time)
    print()




    # --- Generate Causal Questions (Module 2) ---

    generate_questions_start = time.time()
    generate_causal_questions(model, domain, slice_name)
    generate_questions_time = time.time() - generate_questions_start
    print("Causal Question Generation Time: ", generate_questions_time)
    print()


    # --- Generate Causal Statements (Module 3) ---

    generate_statements_start = time.time()
    generate_causal_statements(model, domain, slice_name)
    generate_statements_time = time.time() - generate_statements_start
    print("Causal Statement Generation Time: ", generate_statements_time)
    print()




    # --- Extract Triplets (Module 4) ---
    triple_extraction_start = time.time()
    extract_statement_triples(slice_name)
    triple_extraction_time = time.time() - triple_extraction_start
    print("Causal Triplets Extraction Time: ", triple_extraction_time)
    print()



