Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]Loading checkpoint shards:  25%|██▌       | 1/4 [00:24<01:14, 24.91s/it]Loading checkpoint shards:  50%|█████     | 2/4 [00:50<00:51, 25.59s/it]Loading checkpoint shards:  75%|███████▌  | 3/4 [01:17<00:26, 26.10s/it]Loading checkpoint shards: 100%|██████████| 4/4 [01:23<00:00, 17.90s/it]Loading checkpoint shards: 100%|██████████| 4/4 [01:23<00:00, 20.75s/it]
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Traceback (most recent call last):
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain/chains/query_constructor/base.py", line 52, in parse
    parsed = parse_and_check_json_markdown(text, expected_keys)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/utils/json.py", line 188, in parse_and_check_json_markdown
    raise OutputParserException(
langchain_core.exceptions.OutputParserException: Got invalid return object. Expected key `query` to be present, but got {'affected': [{'defaultStatus': 'unaffected', 'product': 'PAN-OS', 'vendor': 'Palo Alto Networks', 'versions': [{'changes': [{'at': '8.1.24', 'status': 'unaffected'}]}]}]}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 894, in <module>
    main()
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 79, in main
    rag(
  File "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py", line 173, in rag
    ).invoke(standalone_question)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 2876, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/retrievers.py", line 251, in invoke
    raise e
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/retrievers.py", line 244, in invoke
    result = self._get_relevant_documents(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain/retrievers/self_query/base.py", line 271, in _get_relevant_documents
    structured_query = self.query_constructor.invoke(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 2878, in invoke
    input = context.run(step.invoke, input, config)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/output_parsers/base.py", line 192, in invoke
    return self._call_with_config(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/base.py", line 1785, in _call_with_config
    context.run(
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/runnables/config.py", line 427, in call_func_with_variable_args
    return func(input, **kwargs)  # type: ignore[call-arg]
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/output_parsers/base.py", line 193, in <lambda>
    lambda inner_input: self.parse_result([Generation(text=inner_input)]),
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain_core/output_parsers/base.py", line 237, in parse_result
    return self.parse(result[0].text)
  File "/home/gtl1500/miniconda3/envs/personalRAG/lib/python3.10/site-packages/langchain/chains/query_constructor/base.py", line 65, in parse
    raise OutputParserException(
langchain_core.exceptions.OutputParserException: Parsing text
{"affected": [{"defaultStatus": "unaffected", "product": "PAN-OS", "vendor": "Palo Alto Networks", "versions": [{"changes": [{"at": "8.1.24", "status": "unaffected"}
 raised following error:
Got invalid return object. Expected key `query` to be present, but got {'affected': [{'defaultStatus': 'unaffected', 'product': 'PAN-OS', 'vendor': 'Palo Alto Networks', 'versions': [{'changes': [{'at': '8.1.24', 'status': 'unaffected'}]}]}]}
