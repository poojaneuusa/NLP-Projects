# Question-Answer Retrieval
This project implements a Question Answer Retrieval system using the `sentence-transformers` library. The system utilizes semantic search to find relevant answers from a corpus of passages.

### Installation

1. Install the `sentence-transformers` library:
   ```bash
   pip install -U sentence-transformers
   ```

2. Import necessary libraries:
   ```python
   from sentence_transformers import SentenceTransformer, util
   import os
   import json
   import gzip
   ```

3. Download the dataset:
   ```python
   util.http_get('https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/simplewiki-2020-11-01.jsonl.gz', 'simplewiki-2020-11-01.jsonl.gz')
   ```

### Input
- A pre-downloaded dataset (`simplewiki-2020-11-01.jsonl.gz`) containing passages in JSONL format.
- A query string for which the answer is required.

### Process
1. **Dataset Preprocessing:**
   - The dataset is loaded and parsed.
   - Passages are extracted and shuffled, then limited to 100,000 samples.

2. **Model Loading:**
   - A Sentence Transformer model (`nq-distilbert-base-v1`) is loaded for semantic search.

3. **Embedding Creation:**
   - Each passage is converted to a dense vector representation using the model.

4. **Semantic Search:**
   - A query is encoded into a dense vector.
   - The system retrieves the top 3 relevant passages based on cosine similarity between the query vector and passage embeddings.

### Output
- Displays the top 3 relevant passages from the corpus for the given query.

### Example queries
```
get_answer("what is the ML?")
get_answer("when did the first world war end?")
```

### Input
Query: `"when did the first world war end?"`

### Output
```
Results:
['World War I', 'World War I (WWI or WW1), also called the First World War, began on July 28, 1914 and lasted until November 11, 1918. The war was a global war that lasted exactly . Most of the fighting was in Europe, but soldiers from many other countries took part, and it changed the colonial empires of the European powers. Before World War II began in 1939, World War I was called the Great War or the World War. 135 countries took part in World War I, and nearly 10 million people died while fighting.']
['History of the world', 'On November 11, 1918, Germany signed the armistice, meaning "the laying down of arms", to end the war. After the war ended, the Treaty of Versailles was written and Germany was made to sign it. They had to pay $33 million in reparations (payment for damage). The influenza pandemic of 1918 spread around the world, killing millions.']
['World War I', "Before the war, European countries had formed alliances with each other to protect themselves. However, by doing this they had divided themselves into two groups. When Archduke Franz Ferdinand of Austria was assassinated on 28 June 1914, Austria-Hungary blamed Serbia and declared war on them. Serbia's ally Russia then declared war on Austria-Hungary. This set off a chain of events in which the two groups of countries declared war on each other. The two sides were the Allied Powers (mainly Russia, France and the British Empire) and the Central Powers (mainly Germany, Austria-Hungary and the Ottoman Empire)."]
```

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Quora Questions Suggester
This project provides a semantic search system for suggesting related questions on Quora, using Sentence Transformers for natural language processing.
### Installation
```bash
   !pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   !pip install -U sentence-transformers
```
### Process
Loads the `quora-distilbert-multilingual` Sentence Transformer model for semantic encoding.

### Input
- Any natural language question asked by a user
- The system runs interactively, and the user can enter queries repeatedly until exiting by typing 'n'

```
Please enter a question: How can i start with ML?
```

### Output
Displays a list of top 5 related questions with similarity scores.
Sample Output

```
[{'corpus_id': 27939, 'score': 0.9046310186386108}, {'corpus_id': 28547, 'score': 0.9018148183822632}, {'corpus_id': 1548, 'score': 0.8984736204147339}, {'corpus_id': 24961, 'score': 0.8957906365394592}, {'corpus_id': 14644, 'score': 0.8950694799423218}]
0.9046310186386108 :  How to learn MATLAB?
0.9018148183822632 :  What is best way to learn java?
0.8984736204147339 :  How can i learn java programming language?
0.8957906365394592 :  Should I start my start up?
0.8950694799423218 :  How did you learn java?

```

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Similar Research Paper Recommendation
By leveraging Sentence-BERT, the system provides a fast and accurate way to perform semantic search across a large corpus of academic papers, helping researchers find relevant literature quickly

### Installation
```bash
 !pip install -U sentence-transformers
```
Dataset: ``` 'https://sbert.net/datasets/emnlp2016-2018.json' ```
### Input

The input consists of a research paper’s **title** and **abstract** that you want to search for similar papers. 
Example input:

```
Title: Applications of big data in emerging management disciplines: A literature review using text mining
Abstract: The importance of data-driven decisions and support is increasing day by day in every management area. The constant access to volume, variety, and veracity of data has made big data an integral part of management studies. New sub-management areas are emerging day by day with the support of big data to drive businesses. This study takes a systematic literature review approach to uncover the emerging management areas supported by big data in contemporary times.
```
### Output

The output of the system will be the most similar papers to the input query, based on their semantic content

```
Title: PubSE: A Hierarchical Model for Publication Extraction from Academic Homepages
Abstract: Despite recent evidence that Microsoft Academic is an extensive source of citation counts for journal article...
```


-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Text Summarization
This project demonstrates how to use sentence embeddings from the `Sentence-Transformers` library and NetworkX for text summarization. It identifies the most central sentences in a document to generate a summary.

### Installation

```bash
pip install lexrank
pip install -U sentence-transformers
pip install networkx
pip install nltk
```

### Process
This approach uses a combination of sentence embeddings and graph centrality to extract key sentences from a document, which can be used for automatic summarization. Transformer model: `all-MiniLM-L6-v2`
### Input

Input is a text document. For ex:
```
Immediately after the verdict, in a statement released through her spokesperson, Amber had said she was ‘sad’ she had ‘lost the case’. The jury had also found Johnny guilty of defamation on one count and ordered him to pay Amber $2 million in damages. However, most legal experts said the case had been vindication for Johnny.
Speaking about it on Today Show, Amber said about the jury, “I don’t blame them. I actually understand. He’s a beloved character and people feel they know him. He’s a fantastic actor.”
The actor also addressed the memes that have been made about her and the hate coming her way on social media through the trial. She said, “I don’t care what one thinks about me or what judgments you want to make about what happened in the privacy of my own home, in my marriage, behind closed doors. I don’t presume the average person should know those things. And so I don’t take it personally. But even somebody who is sure I’m deserving of all this hate and vitriol, even if you think that I’m lying, you still couldn’t look me in the eye and tell me that you think on social media there’s been a fair representation. You cannot tell me that you think that this has been fair.”
```
### Output
The output of this process is the top 3 most central sentences from the document. For example:

```
Most central sentences:
1. Speaking about it on Today Show, Amber said about the jury, “I don’t blame them. I actually understand. He’s a beloved character and people feel they know him. He’s a fantastic actor.”
2. The actor also addressed the memes that have been made about her and the hate coming her way on social media through the trial. She said, “I don’t care what one thinks about me or what judgments you want to make about what happened in the privacy of my own home, in my marriage, behind closed doors. I don’t presume the average person should know those things. And so I don’t take it personally."
3. Immediately after the verdict, in a statement released through her spokesperson, Amber had said she was ‘sad’ she had ‘lost the case’. The jury had also found Johnny guilty of defamation on one count and ordered him to pay Amber $2 million in damages. However, most legal experts said the case had been vindication for Johnny.

```
