All code in this personal project is entirely mine. It was inspired by a group project completed with two other students under INSuRE+C. Some of my code, included in this personal project, was also included in the group project.

A Retrieval Augmented Generator, or RAG, is a type of AI model that can 1) translate a person's input into embeddings (computer language), 2) search through a database for documents with similar embeddings (documents related to the input), 3) pass those documents + the input to a generator (an AI/LLM), which 4) outputs an informed response.

In October 2023, two other students, a professor, and I partnered with Argonne National Laboratory to create a digital cybersecurity advisor tailored toward the power industry. By my suggestion, we accomplished this via a RAG; we used HuggingFace, Langchain, and Llama 3 from Meta. The successful completion of our model will address a fundamental issue faced within the power industry: small teams with few resources will be able to accomplish more with less.

The parts of my code that were included in the group project were [CreateDatabase.py](CreateDatabase.py), as well as the following pieces of [RAG.py](RAG.py): createRetriever() and all of its helper functions and the prompts from cons_prompt() and sc_prompt(). For this personal project, I had to make my own version of the LLM/AI used to generate responses. I wanted to create this personal project for a few different reasons. The primary reason was that I wanted experience in developing every part of the RAG since I was not able to code everything in the group project. Relatedly, I wished to understand the inner workings of the RAG. Finally, I wanted a chance to organize the code to my taste.

[CreateDatabase.py](CreateDatabase.py) can be used to create a Chroma Vector Database from a folder of JSON files. The code can be changed to read other types of files, but I only needed to read JSON files when I created it. I may add more functionality as required.

The main() function of [RAG.py](RAG.py) asks the same question three different times using different prompts - one of these prompts is primarily the creation of LangChain, and the other two are entirely my own. [RAG.o](SLURM/RAG.o) is an example output from running [RAG.py](RAG.py).

This project is a work in progress. Currently, several areas need improvement. In the order that I plan to tackle them, these issues are:

- My model seems to be ignoring the audience specified in my prompts. Llama-based models have a special prompt format. I did not implement this special format in my project because the results I got from nearly identical prompts in the group project were quite good without the special format. However, since I am facing issues, I will implement it here to see if it improves my results. It may even improve the performance of my LLM/AI. Which brings me to my next point...

- The LLM/AI portion of [RAG.py](RAG.py) is not performing as well as the LLM/AI used in the group project. Mine frequently freezes up when I run my script. Because of the LLM/AI from the project, I KNOW that better results are possible. This presents an opportunity for me to become better acquainted with different LLM/AI parameters.

- I've realized that LangChain has a RecursiveJsonSplitter. In my [CreateDatabase.py](CreateDatabase.py), I used a CharacterTextSplitter instead, which I believe is wrong. I noticed that my document "chunks" did not seem to be 250 tokens long, and I think this is why. Fixing this issue will also require me to improve the query_constructor_prompt() I have in [RAG.py](RAG.py).

- I have not been able to get chat history working. This comes down to me not understanding the mechanics behind LangChain's concept of chat history. Implementing it shouldn't be too tricky, but I'd prefer to improve my model's performance without chat history before adding that extra variable.

- Creating proper classes with corresponding methods could be good for organization and functionality, but I think that would require a lot of refactoring, so it is the lowest on my list.

I will work on these issues in a separate branch and leave this semi-functional code in the main branch.