Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]Loading checkpoint shards:  25%|██▌       | 1/4 [00:07<00:22,  7.62s/it]Loading checkpoint shards:  50%|█████     | 2/4 [00:15<00:15,  7.54s/it]Loading checkpoint shards:  75%|███████▌  | 3/4 [00:22<00:07,  7.34s/it]Loading checkpoint shards: 100%|██████████| 4/4 [00:23<00:00,  5.13s/it]Loading checkpoint shards: 100%|██████████| 4/4 [00:23<00:00,  5.99s/it]
Setting the `device` argument to None from -1 to avoid the error caused by attempting to move the model that was already loaded on the GPU using the Accelerate module to the same or another device.
Traceback (most recent call last):
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 813, in <module>
    main()
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 77, in main
    retriever = create_retriever( # initialize retriever
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 243, in create_retriever
    query_constructor = json_query_generator(llm),
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 353, in json_query_generator
    query_constructor_prompt() # see query_constructor_prompt() under PROMPTS
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 608, in query_constructor_prompt
    examples = query_constructor_example(
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 520, in query_constructor_example
    example = f"""
ValueError: Invalid format specifier
