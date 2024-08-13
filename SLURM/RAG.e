Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]Loading checkpoint shards:  25%|██▌       | 1/4 [00:23<01:11, 23.75s/it]Loading checkpoint shards:  50%|█████     | 2/4 [00:50<00:50, 25.34s/it]Loading checkpoint shards:  75%|███████▌  | 3/4 [00:57<00:17, 17.13s/it]Loading checkpoint shards: 100%|██████████| 4/4 [00:59<00:00, 11.12s/it]Loading checkpoint shards: 100%|██████████| 4/4 [00:59<00:00, 14.87s/it]
Setting the `device` argument to None from -1 to avoid the error caused by attempting to move the model that was already loaded on the GPU using the Accelerate module to the same or another device.
Traceback (most recent call last):
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 576, in <module>
    main()
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 85, in main
    chat_history = rag( # rag prints output and stores input and output in chat history
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 146, in rag
    task_prompt = task_chain( # prompt to describe what response the llm should generate
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 320, in task_chain
    variables
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
TypeError: Expected a Runnable, callable or dict.Instead got an unsupported type: <class 'str'>
