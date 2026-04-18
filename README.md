# LCM

This repository holds the code for reproducing the results of the paper "Large Causal Models from Large Language Models"
- https://arxiv.org/abs/2512.07796

## Requirements

This pipeline requires the huggingface transformers libary, pytorch, sentence_transformers, and scikit_learn. As is, this uses the "mistralai/Mistral-Small-24B-Instruct-2501" model.

The triplet extraction section of this pipeline also depends to the python libary triplet-extract and spacy. See here to install: https://github.com/adlumal/triplet-extract, or install with:

```
pip install triplet-extract[deepsearch]
python -m spacy download en_core_web_sm
```