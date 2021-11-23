---
title: SemanticSearch HU

emoji: 💻

colorFrom: green

colorTo: white

sdk: streamlit

app_file: src/app.py

pinned: false

---

# Huggingface Course Project - 2021 November
## Semantic Search system in Hungarian

This repo contains my course project created during the week of the Huggingface Course Launch Community event. The selected project is a denser retrieval based semantic search system in my own language, Hungarian. It is based on [this question-answering project idea description](https://discuss.huggingface.co/t/build-a-question-answering-system-in-your-own-language/11570/2).

## Approach
- finding a **dataset** of question/answer pairs or descriptive paragraphs in my target language (Hungarian)
- using a **pretrained model** (Hungarian or multilingual) to generate embeddings for all answers, preferably using sentence-transformers
- **search for top-K matches** - when user query is entered, generate the query embedding and search through all the answer embeddings fo find the top-K most likely documents

## Dataset - raw text

Two datasets were evaluated:
1. [not used] [MQA - multilingual Question-Answering](https://huggingface.co/datasets/clips/mqa), with a Hungarian subset

This datasets contains two types of data:
* FAQ, about 800.000 questions and answers scraped from different websites (Common Crawl). The problem with this dataset is that it only contains text from roughly 2.000 different domains (so many of the questions and answers are repetitive), and also the quality of the answers varies greatly, for some domains it is not really relevant (for example full of url references).
* CQA, about 27.000 community question answering examples, which were scraped from different forums. Here for every questions there are several answers, but again the quality of the answers varies greatly, with many answers not being relevant.

2. **[used] [DBpedia - short abstracts in Hungarian](https://databus.dbpedia.org/dbpedia/text/short-abstracts)**

This data contains 450.000 shortened abstract from Wikipedia in Hungarian. This represents the text before the table of contents of Wikipedia articles, shortened to approximately 2-3 sentences. These texts seemed like high quality paragraphs, and so I decided to use them as a bank of "answers".

The format of the data is of RDF Turtle (Resource Description Framework), which is a rich format to relate metadata and model information. In our case, we just want to use a fraction of this data, only the pure text of each abstract. The raw text was extracted using `rdflib` library seen in the script in `src/data/dbpedia_dump_wiki_text.py`.

## Model - precalculate embeddings

To generate the embeddings for each paragraph/shortened abstract, a sentence embedding approach was used. [SBERT.net](https://www.sbert.net/index.html) offers a framework and lots of pretrained models in more than 100 languages to create embeddings and compare them, to find the ones with similar meaning.

This task is also called STS (Semantic Text Similarity) or Semantic Search, which seeks to find similarity not just based on lexical matches, but by comparing vector representations of the content and thus improving accuracy. 

There were various [pretrained models](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models) to choose from. For this project the **`paraphrase-multilingual-MiniLM-L12-v2`** checkpoint is used, as this is one of the smallest multilingual models at 418 MB, but it has the second fastest encoding speed, which seems like a good compromise.

```
Model facts:
- Checkpoint name: paraphrase-multilingual-MiniLM-L12-v2 
- Dimensions: 384
- Suitable Score Functions: cosine-similarity
- Pooling: Mean Pooling
```

- Embeddings were calculated based on code examples from [huggingface hub](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Similarity scores were calculated based on code example from [sentence-transformers site](https://www.sbert.net/examples/applications/semantic-search/README.html)

To reproduce the precalculated embedding use the notebook in `notebooks/QA_retrieval_precalculate_embeddings.ipynb`, with GPU in Google Colab.

Known bug: the precalculated embeddings contain an extra tensor at the end, which is the empty newline at the end of the text file, this last index should be ignored

## Search top-k matches

Finally, having all precalculated embeddings, we can to implement semantic search (dense retrieval).We encode the search query into vector space and retrieves the document embeddings that are closest in vector space (using cosine similarity). By default the top 5 similar wikipedia abstracts are returned. Can be seen in the main script `src/main_qa.py`.
