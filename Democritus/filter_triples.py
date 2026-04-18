import re
import json
from collections import deque
import time
import sys
 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
 
 
CAUSAL_RELATIONS = {
    "causes", "cause",
    "leads to", "lead to"
}

# Words that should never appear alone as a subject or object
STOPWORDS = {
    # Pronouns
    'it', 'this', 'that', 'they', 'them', 'we', 'he', 'she', 'you', 'i',
    'which', 'what', 'who', 'there', 'here', 'these', 'those', 'its',
    # Generic programming terms that appear as extraction artifacts
    'class', 'object', 'instance', 'method', 'function', 'value',
    'type', 'variable', 'parameter', 'result', 'output', 'input',
    # Articles / determiners
    'the', 'a', 'an',
    # Misc
    'etc', 'example', 'case', 'way', 'thing', 'things',
}

SUB_OBJ_LEN_MIN = 5 #Minimum subject and object length

EMBED_SIMILARITY_THRESHOLD = 92 # Similarity threshold for cosine similarity after embedding
FILTER_WINDOW = 20 # Number of triples checked against when iterating through file



# ---------------------------------------------------------------------------
# Stage 1: Normalization
# ---------------------------------------------------------------------------
 
def normalize_span(text: str) -> str:
    """
    Clean a subject or object span:
      - Strip leading bullet/dash artifacts (e.g. "-Insufficient test automation")
      - Strip leading articles
      - Normalize whitespace
      - Strip trailing punctuation
    """
    text = text.strip()
    # Strip leading dash/bullet artifacts from LLM bullet list formatting
    text = re.sub(r'^[-•*–—]+\s*', '', text)
    # Strip leading articles
    text = re.sub(
        r'^(a|an|the|this|that|these|those|its|their|such|certain)\s+',
        '', text, flags=re.IGNORECASE
    )
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Strip trailing punctuation
    text = text.strip('.,;:\'\"')
    return text
 
 
def normalize_triple(triple: dict) -> dict:
    """Apply normalization to subject and object of a triple."""
    triple = dict(triple)  # don't mutate original
    triple['subject'] = normalize_span(triple['subject'])
    triple['object']  = normalize_span(triple['object'])
    triple['relation'] = triple['relation'].strip().lower()
    return triple




# ---------------------------------------------------------------------------
# Stage 2: Validity filter
# ---------------------------------------------------------------------------

def is_valid_triple(triple):
    subj = triple['subject'].strip()
    obj  = triple['object'].strip()
    rel  = triple['relation'].strip()

    # Too short
    if len(subj) < SUB_OBJ_LEN_MIN or len(obj) < SUB_OBJ_LEN_MIN:
        return False

    # Subject or object is a stopword or pronoun
    if subj.lower() in STOPWORDS or obj.lower() in STOPWORDS:
        return False

    # Subject equals object
    if subj.lower() == obj.lower():
        return False

    # Non-causal relation slipping through
    if rel.lower() not in CAUSAL_RELATIONS:
        return False

    # Leading dash artifact from LLM bullet formatting
    # (your "-Insufficient test automation" example)
    triple['subject'] = subj.lstrip('-').strip()

    return True


def filter_keywords(triple_file):
    triple_count = 0
    valid_count = 0
    valid_triples = []
    with open(triple_file, 'r') as f:
        for line in f:
            triple_count += 1

            # Verify triple 
            triple = json.loads(line)
            triple = normalize_triple(triple)
            if is_valid_triple(triple):
                valid_triples.append(triple)
                valid_count += 1
    
    return valid_triples, triple_count, valid_count
                

# ---------------------------------------------------------------------------
# Stage 2: Duplicate/Subset filter
# ---------------------------------------------------------------------------

from collections import deque
import re

def to_words(text: str) -> list[str]:
    """Split text into words, treating hyphens as separators."""
    return re.split(r'[\s\-,;]+', text.lower().strip())

def is_subsequence(shorter_words: list[str], longer_words: list[str]) -> bool:
    """Return True if all words in shorter_words appear in longer_words in the same order."""
    longer_iter = iter(longer_words)
    return all(word in longer_iter for word in shorter_words)

def is_degraded_span(candidate: str, reference: str) -> bool:
    """
    Return True if candidate is a word-level subsequence of reference,
    meaning candidate is a shortened/degraded version of reference.
    e.g. "compile error detection" is a degraded span of "compile-time error detection"
    """
    candidate_words = to_words(candidate)
    reference_words = to_words(reference)
    if candidate_words == reference_words:
        return False
    return is_subsequence(candidate_words, reference_words)

def remove_redundant_triples(
    triples: list[dict],
    model: SentenceTransformer,
    similarity_threshold: float = 0.92,
    window_size: int = 20,
) -> list[dict]:
    """
    Remove redundant triples using a sliding window queue.
    A triple is considered redundant if:
      - Its subject is a degraded version of an already-queued subject, AND
      - Its object is similar to the queued object (by subsequence or embedding)
    When a queued triple is found to be degraded relative to a new arrival,
    the queued triple is replaced by the more informative new one.
    Triples age out of the queue into the final list after window_size steps.
    """
    pending_queue = deque()       # triples still within comparison window
    pending_obj_embs = deque()    # object embeddings parallel to pending_queue
    final_triples = []
    removed_count = 0

    for current_triple in triples:
        current_subject = current_triple['subject'].lower()
        current_object  = current_triple['object'].lower()
        current_obj_emb = model.encode(
            [current_triple['object']], show_progress_bar=False
        )[0]

        # Convert deques to lists for index-based manipulation
        queue_as_list    = list(pending_queue)
        obj_embs_as_list = list(pending_obj_embs)

        dominated_queue_indices = []
        current_is_redundant = False

        for queue_idx, queued_triple in enumerate(queue_as_list):
            if current_triple['relation'] != queued_triple['relation']:
                continue

            queued_subject = queued_triple['subject'].lower()
            queued_object  = queued_triple['object'].lower()
            queued_obj_emb = obj_embs_as_list[queue_idx]

            # Check if objects are similar enough to consider subjects comparable
            obj_similarity = cosine_similarity(
                [current_obj_emb], [queued_obj_emb]
            )[0][0]
            objects_are_similar = (
                current_object == queued_object
                or is_degraded_span(current_object, queued_object)
                or is_degraded_span(queued_object, current_object)
                or obj_similarity > similarity_threshold
            )

            if not objects_are_similar:
                continue

            # Current subject is a degraded version of queued — current is worse
            if is_degraded_span(current_subject, queued_subject):
                current_is_redundant = True
                break

            # Queued subject is a degraded version of current — queued is worse
            if is_degraded_span(queued_subject, current_subject):
                dominated_queue_indices.append(queue_idx)

            # Identical subjects — keep whichever has the more informative object
            if current_subject == queued_subject:
                if len(current_object) >= len(queued_object):
                    dominated_queue_indices.append(queue_idx)
                else:
                    current_is_redundant = True
                    break

        # Remove dominated queue entries from back to front to preserve indices
        for queue_idx in sorted(dominated_queue_indices, reverse=True):
            del queue_as_list[queue_idx]
            del obj_embs_as_list[queue_idx]
            removed_count += 1

        # Rebuild deques from modified lists
        pending_queue    = deque(queue_as_list)
        pending_obj_embs = deque(obj_embs_as_list)

        if not current_is_redundant:
            pending_queue.append(current_triple)
            pending_obj_embs.append(current_obj_emb)
        else:
            removed_count += 1

        # Evict oldest entry once queue exceeds window size
        if len(pending_queue) > window_size:
            final_triples.append(pending_queue.popleft())
            pending_obj_embs.popleft()

    # Flush remaining queue entries to final list
    final_triples.extend(pending_queue)

    return final_triples


def filter_duplicate_triples(triples):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    max_triples = remove_redundant_triples(triples, model, similarity_threshold=EMBED_SIMILARITY_THRESHOLD, window_size=FILTER_WINDOW)
    return max_triples
 



def filter_file(file):

    print(f"Filtering File: {file}")
    in_file = "Causal_Triples/" + file
    out_file = "Filtered_Causal_Triples/" + file

    start_time = time.time()

    key_filtered_triples, total_triples, total_key_triples_removed = filter_keywords(in_file)

    dup_filtered_triples = filter_duplicate_triples(key_filtered_triples)

    with open(out_file, 'w') as file:
        for triple in dup_filtered_triples:
            file.write(json.dumps(triple) + '\n')
    
    # print(total_triples)
    # print(total_key_triples_removed)
    # print(len(dup_filtered_triples))

    
    triples_removed = total_triples - len(dup_filtered_triples)
    print(f"Removed {triples_removed} out of {total_triples} triples")
    print(f"Runtime: {time.time()-start_time} seconds")


def main():
    file_list = [
        "triples_swe_depth1.jsonl",
        "triples_swe_depth2.jsonl",
        "triples_swe_depth3.jsonl",
        "triples_swe_depth4.jsonl",
    ]

    for file in file_list:
        filter_file(file)
        sys.stdout.flush()

    


 
 
if __name__ == "__main__":
    main()