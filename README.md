## Learning Inflation Narratives from Reddit: How Lightweight LLMs Reveal Forward-Looking Economic Signals

This repository contains the code and data associated with our peer-reviewing paper, "Learning Inflation Narratives from Reddit: How Lightweight LLMs Reveal Forward-Looking Economic Signals".

This project introduces a novel Natural Language Processing (NLP) approach to measure public perception of inflation using lightweight large language models (LLMs). We demonstrate how social media discussions can serve as a valuable, forward-looking economic signal, offering insights into public sentiment that complement traditional economic indicators.

### Contributions:
1.	We design an observational and training dataset that links Reddit posts to the inflation indicator’s components and filters them based on price-related keywords.
2.	We show that fine-tuning lightweight LLMs with domain-adaptive data results in a robust inflation index that aligns closely with, and at times leads, official economic statistics.
3.	We provide a qualitative analysis of the thematic narratives underlying public inflation sentiment, revealing specific concerns around food, real estate, and travel that offer deep insights. 

Read the full paper here:

### Repository Structure
- ``notebooks``: Mainly include codes for fine-tuning the large language models to create the inflation classifier on Google Colaboratory.
- ``src``: Mainly include Python codes for data collection from the Reddit archive and statistical analysis.
- ``data``
