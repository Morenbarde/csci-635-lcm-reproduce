import argparse
from Democritus import pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DEMOCRITUS pipeline for a given domain and topic depth"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=1,
        help="Topic expansion depth (default: 1)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="Output/",
        help="Output folder for generated files (default: Output/)"
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

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

    # domain = "neuroscience and medicine"
    # root_topics = [ "Neuroscience",
    #                 "Genetics",
    #                 "Evolution",
    #                 "Botany",
    #                 "Cardiology",
    #                 "Endocrinology",
    #                 "Immunology",
    #                 "Oncology",
    #                 "Exercise physiology",
    #                 "Metabolic disorders"]

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

    domain = "software engineering and computer science"

    root_topics = [
        "Software Design and Architecture",
        "Software Testing and Quality Assurance",
        "DevOps and Continuous Integration",
        "Database Systems and Data Management",
        "Computer Networks and Distributed Systems",
        "Cybersecurity and Secure Software Development",
        "Human-Computer Interaction and Usability",
        "Artificial Intelligence and Machine Learning Engineering",
        "Software Project Management and Agile Methods",
        "Programming Languages and Compilers",
    ]

    topic_depth = args.depth
    slice_name  = "swe_depth" + str(topic_depth)
    output_folder = args.output

    pipeline.run_full(domain, root_topics, topic_depth, slice_name, output_folder)