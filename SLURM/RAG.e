Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]Loading checkpoint shards:  25%|██▌       | 1/4 [00:43<02:11, 43.93s/it]Loading checkpoint shards:  50%|█████     | 2/4 [01:28<01:29, 44.52s/it]Loading checkpoint shards:  75%|███████▌  | 3/4 [02:12<00:44, 44.13s/it]Loading checkpoint shards: 100%|██████████| 4/4 [02:21<00:00, 30.10s/it]Loading checkpoint shards: 100%|██████████| 4/4 [02:21<00:00, 35.28s/it]
Setting the `device` argument to None from -1 to avoid the error caused by attempting to move the model that was already loaded on the GPU using the Accelerate module to the same or another device.
Traceback (most recent call last):
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 875, in <module>
    main()
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 57, in main
    rag(
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 192, in rag
    print(chain.invoke(standalone_question))
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 2876, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 3580, in invoke
    output = {key: future.result() for key, future in zip(steps, futures)}
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 3580, in <dictcomp>
    output = {key: future.result() for key, future in zip(steps, futures)}
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/concurrent/futures/_base.py", line 445, in result
    return self.__get_result()
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/concurrent/futures/_base.py", line 390, in __get_result
    raise self._exception
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/concurrent/futures/thread.py", line 52, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 3564, in _invoke_step
    return context.run(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 2876, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/retrievers.py", line 251, in invoke
    raise e
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/retrievers.py", line 244, in invoke
    result = self._get_relevant_documents(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain/retrievers/self_query/base.py", line 271, in _get_relevant_documents
    structured_query = self.query_constructor.invoke(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 2876, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 3580, in invoke
    output = {key: future.result() for key, future in zip(steps, futures)}
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 3580, in <dictcomp>
    output = {key: future.result() for key, future in zip(steps, futures)}
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/concurrent/futures/_base.py", line 445, in result
    return self.__get_result()
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/concurrent/futures/_base.py", line 390, in __get_result
    raise self._exception
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/concurrent/futures/thread.py", line 52, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 3564, in _invoke_step
    return context.run(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 2878, in invoke
    input = context.run(step.invoke, input, config)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/language_models/llms.py", line 344, in invoke
    self.generate_prompt(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/language_models/llms.py", line 701, in generate_prompt
    return self.generate(prompt_strings, stop=stop, callbacks=callbacks, **kwargs)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/language_models/llms.py", line 880, in generate
    output = self._generate_helper(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/language_models/llms.py", line 738, in _generate_helper
    raise e
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/language_models/llms.py", line 725, in _generate_helper
    self._generate(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_huggingface/llms/huggingface_pipeline.py", line 269, in _generate
    responses = self.pipeline(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/transformers/pipelines/text_generation.py", line 262, in __call__
    return super().__call__(text_inputs, **kwargs)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/transformers/pipelines/base.py", line 1238, in __call__
    outputs = list(final_iterator)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/transformers/pipelines/pt_utils.py", line 124, in __next__
    item = next(self.iterator)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/transformers/pipelines/pt_utils.py", line 125, in __next__
    processed = self.infer(item, **self.params)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/transformers/pipelines/base.py", line 1164, in forward
    model_outputs = self._forward(model_inputs, **forward_params)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/transformers/pipelines/text_generation.py", line 351, in _forward
    generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/transformers/generation/utils.py", line 2024, in generate
    result = self._sample(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/transformers/generation/utils.py", line 3020, in _sample
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
