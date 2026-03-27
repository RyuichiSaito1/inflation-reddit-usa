## Learning Inflation Narratives from Reddit: How Lightweight LLMs Reveal Forward-Looking Economic Signals

This repository contains the code and data associated with our peer-reviewing paper, "Learning Inflation Narratives from Reddit: How Lightweight LLMs Reveal Forward-Looking Economic Signals".

This project introduces a novel Natural Language Processing (NLP) approach to measure public perception of inflation using lightweight large language models (LLMs). We demonstrate how social media discussions can serve as a valuable, forward-looking economic signal, offering insights into public sentiment that complement traditional economic surveys.

### Contributions:
1.	We design CPI-component-aligned subreddit inputs and build observational and training datasets for category-level inflation signal extraction.
2.	We show that fine-tuning lightweight LLMs with domain-adaptive data results in a robust inflation index that aligns closely with, and at times precedes, official economic statistics.
3.	We provide a qualitative analysis of the thematic narratives underlying public inflation sentiment, revealing specific concerns around food, cars, real estate, travel, and frugality that offer deep insights.

Read the full paper here: https://arxiv.org/abs/2603.21501

### Repository Structure
- ``notebooks``: include codes for fine-tuning the large language models to create the inflation classifier on Google Colaboratory.
- ``src``: include Python codes for data collection from the Reddit archive and statistical analysis to validate results.
- ``data``: include metadata for collection datasets in which Reddit users' personal information was excluded.
