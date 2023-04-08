# Question Answering with Transformers

The purpose of this project was the development of a question answering engine over the Wikidata knowledge graph. The implementation leverages the Bidirectional Encoder Representations from Transformers (BERT) model to perform Span Entity and Relation prediction on questions posed in natural English. An overview can be seen below.

![blueprint](https://user-images.githubusercontent.com/73662635/230367368-e4b43fbd-a0a9-4d82-9f32-390f95ef30bd.png)

1. A simple interface that allows the user to pose a natural language question.
2. Modules for preprocessing and converting into token IDs and attention masks.
3. BERT-based relation and span entity prediction models.
4. A query generator that creates a SPARQL query equivalent to the user's question.
5. A query executor that retrieves the answer in natural language from the Wikidata endpoint.

The integration of these components resulted in a highly effective question-answering engine, which is demonstrated by its impressive performance on the test set as well as on unseen data.

## Examples

![examples](https://user-images.githubusercontent.com/73662635/230643611-0131bcb3-e6fa-488c-b693-7bb662e1e509.png)

The questions above feature entities that were not part of the training dataset or the dictionary and demonstrate the model's ability to generalize well and provide accurate answers.

## Notebooks

### [Dataset](01_dataset.ipynb)

The first notebook contains the code for data ingestion of the Simple Questions datasets. The Wikidata client was used to retrieve the labels for both the entity and relation identifiers. The spacy library was utilized when identifying semantically similar entities was necessary during the entity mapping process. Preprocessing of the questions included removing accents, special characters and possessive suffixes to improve entity identification. 

### [Experiments](02_experiments.ipynb)

In the second notebook are my initial experiments with single model architectures for both span entity and relation prediction. This approach didn't prove fruitful as the model struggled to learn both tasks effectively. Possible reasons for this include a bias towards learning entity spans during training and the complexity of the relation prediction task.

### [Models](03_models.ipynb)

The models notebook contains the implementation of separate span entity and relation models, which proved to be effective in addressing the limitations of the single model approach. Both showed excellent performance, with the span prediction model achieving 87.52% Dataset Wide F1-score and the relation prediction model 88.55% accuracy on the test set. 

### [Engine](04_engine.ipynb)

The final notebook showcases the development of the question-answering engine modules. Once the entity prediction is made, an initial query is constructed to obtain its ID. If this query is unsuccessful, the system searches for the closest match. The builder and executor modules adhere to the SPARQL documentation. You can explore the examples in this notebook or run the [enquire.py](enquire.py) file from the terminal to experiment further.

### References

1. [Pretrained Transformers for Simple Question Answering over Knowledge Graphs](https://arxiv.org/pdf/2001.11985.pdf) by Denis Lukovnikov, Asja Fischer and Jens Lehmann.
2. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
3. [Question Answering Benchmarks for Wikidata](https://ceur-ws.org/Vol-1963/paper555.pdf) by Dennis Diefenbach, Thomas Pellissier Tanon, Kamal Singh and Pierre Maret.
4. [Simple Questions](https://github.com/askplatypus/wikidata-simplequestions) repository with mapping of Freebase topics to Wikidata items.
