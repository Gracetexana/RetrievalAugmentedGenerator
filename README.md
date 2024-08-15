# RetrievalAugmentedGenerator
This code is INSPIRED by a group project completed with two other students under INSuRE+C. All code within this personal project is entirely mine. Some of my code, included in this personal project, was included in the group project as well.

A Retrieval Augmented Generator, or RAG, is a type of AI model that can 1) translate a person's input into embeddings (computer language), 2) search through a database for documents with similar embeddings (documents related to the input), 3) pass those documents + the input to a generator (an AI/LLM), which 4) outputs an informed response to the input.

In October 2023, two other students, a professor, and I partnered with Argonne National Laboratory to create a digital cybersecurity advisor tailored towards the power industry. By my suggestion, we opted to accomplish this via a RAG; we used HuggingFace, Langchain, and Llama 3 from Meta. The successful completion of our model will address a real issue faced within the power industry: small teams with few resources will be able to accomplish more with less.

The parts of my code that were included in the group project were [CreateDatabase.py](CreateDatabase.py), as well as the following pieces of [RAG.py](RAG.py): createRetriever() and all of its helper functions and the prompts from cons_prompt() and sc_prompt(). I wanted to create my own project for a few different reasons. The primary reason was that I wanted experience in creating every part of the RAG since I was not able to code everything in the group project. Relatedly, I wanted to understand the inner workings of the RAG. Finally, I felt that the code from our project was not very organized, so I wanted to make my own organized version.

[CreateDatabase.py](CreateDatabase.py) can be used to create a Chroma Vector Database from a folder of JSON files. The code can be changed to read other types of files, but when I created it, I only needed to read JSON files. I may add more functionality as needed.

The main() function of [RAG.py](RAG.py) has a few example questions showcasing the abilities of the RAG.