from model import Democritus_Model
from topic_graph import generate_topic_graph
from causal_generation import generate_causal_questions, generate_causal_statements
from triplet_extraction import extract_statement_triples

import time


if __name__ == "__main__":

    # domain = "macroeconomics and financial markets"
    # root_topics = [ "Macroeconomics",
    #                 "Microeconomics",
    #                 "Game Theory",
    #                 "Finance",
    #                 "Trade",
    #                 "Marketing",
    #                 "Stock Market",
    #                 "Investing",
    #                 "Cryptocurrency",
    #                 "Bonds",
    #                 "Monetary Policy",
    #                 "Banking",
    #                 "Fiscal Policy",
    #                 "Inflation",
    #                 "Unemployment"]

    domain = "neuroscience and medicine"
    root_topics = [ "Neuroscience",
                    "Genetics",
                    "Evolution",
                    "Botany",
                    "Cardiology",
                    "Endocrinology",
                    "Immunology",
                    "Oncology",
                    "Exercise physiology",
                    "Metabolic disorders"]

    # domain = "South Asian archaeology and paleoclimate"
    # root_topics = [ "Indus Valley Civilization",
    #                 "Harappan urban centers (Harappa, Mohenjo-daro, Dholavira)",
    #                 "Mohenjo-daro urban planning and sanitation systems",
    #                 "Indus script and undeciphered writing systems",
    #                 "Epigraphy and decipherment of ancient scripts",
    #                 "Holocene monsoon variability in South Asia",
    #                 "4.2 ka event and global Bronze Age disruptions",
    #                 "Climate-induced crop shifts and agricultural adaptation strategies",
    #                 "Irrigation and agriculture in semi-arid river basins",
    #                 "Floodplain farming along the Indus and its tributaries"]
    
    slice_name = "bio_depth0" # For File Naming
    topic_depth = 0


    # --- Load Model ---

    load_model_start = time.time()
    model = Democritus_Model()
    model_load_time = time.time() - load_model_start
    print("Model Load Time: ", model_load_time)
    print()


    # --- Generate Topic Graph (Module 1) ---

    generate_topics_start = time.time()
    generate_topic_graph(model, domain, root_topics, topic_depth, slice_name)
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



