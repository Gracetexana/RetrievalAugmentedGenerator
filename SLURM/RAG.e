Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]Loading checkpoint shards:  25%|██▌       | 1/4 [00:27<01:22, 27.40s/it]Loading checkpoint shards:  50%|█████     | 2/4 [00:56<00:56, 28.37s/it]Loading checkpoint shards:  75%|███████▌  | 3/4 [01:23<00:27, 27.97s/it]Loading checkpoint shards: 100%|██████████| 4/4 [01:29<00:00, 19.21s/it]Loading checkpoint shards: 100%|██████████| 4/4 [01:29<00:00, 22.43s/it]
Setting the `device` argument to None from -1 to avoid the error caused by attempting to move the model that was already loaded on the GPU using the Accelerate module to the same or another device.
Traceback (most recent call last):
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 838, in <module>
    main()
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 93, in main
    chat_history = rag( # testing chat history
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 142, in rag
    standalone_question = standalone_question_generator(llm, chat_history).invoke(question) # standalone_question_generator() is under GENERATORS
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 327, in standalone_question_generator
    {"chat_history" : chat_history, "question" : RunnablePassthrough()} # chat history and the original question...
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 448, in __ror__
    return RunnableSequence(coerce_to_runnable(other), self)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 5552, in coerce_to_runnable
    return cast(Runnable[Input, Output], RunnableParallel(thing))
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 3394, in __init__
    steps__={key: coerce_to_runnable(r) for key, r in merged.items()}
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 3394, in <dictcomp>
    steps__={key: coerce_to_runnable(r) for key, r in merged.items()}
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 5554, in coerce_to_runnable
    raise TypeError(
TypeError: Expected a Runnable, callable or dict.Instead got an unsupported type: <class 'list'>
