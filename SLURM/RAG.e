Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]Loading checkpoint shards:  25%|██▌       | 1/4 [00:27<01:21, 27.01s/it]Loading checkpoint shards:  50%|█████     | 2/4 [00:54<00:54, 27.46s/it]Loading checkpoint shards:  75%|███████▌  | 3/4 [01:02<00:18, 18.53s/it]Loading checkpoint shards: 100%|██████████| 4/4 [01:05<00:00, 12.32s/it]Loading checkpoint shards: 100%|██████████| 4/4 [01:05<00:00, 16.37s/it]
Setting the `device` argument to None from -1 to avoid the error caused by attempting to move the model that was already loaded on the GPU using the Accelerate module to the same or another device.
Traceback (most recent call last):
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 855, in <module>
    main()
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 86, in main
    chat_history = rag( # rag prints output and stores input and output in chat history
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 145, in rag
    context = format_docs(retriever.invoke(standalone_question)) # the documents that the LLM will use to answer the question
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/retrievers.py", line 251, in invoke
    raise e
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/retrievers.py", line 244, in invoke
    result = self._get_relevant_documents(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain/retrievers/self_query/base.py", line 276, in _get_relevant_documents
    new_query, search_kwargs = self._prepare_query(query, structured_query)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain/retrievers/self_query/base.py", line 238, in _prepare_query
    new_query, new_kwargs = self.structured_query_translator.visit_structured_query(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_community/query_constructors/chroma.py", line 49, in visit_structured_query
    kwargs = {"filter": structured_query.filter.accept(self)}
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/structured_query.py", line 82, in accept
    return getattr(visitor, f"visit_{_to_snake_case(self.__class__.__name__)}")(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_community/query_constructors/chroma.py", line 34, in visit_operation
    return {self._format_func(operation.operator): args}
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_community/query_constructors/chroma.py", line 29, in _format_func
    self._validate_func(func)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/structured_query.py", line 23, in _validate_func
    raise ValueError(
ValueError: Received disallowed operator not. Allowed comparators are [<Operator.AND: 'and'>, <Operator.OR: 'or'>]
