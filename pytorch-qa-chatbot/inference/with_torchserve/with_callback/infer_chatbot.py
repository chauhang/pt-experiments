import logging
logger = logging.getLogger(__name__)


# def parse_response(text):
#     return_msg = ""
#     if "### Response:" in text:
#         response_text = text.split("### Response:")[-1]
#         response_text = response_text.split("###")[0]
#         return_msg += response_text
#
#         if "### Example" in text:
#             example_text = text.split("### Example")[-1]
#             example_text = example_text.split("###")[0]
#             example_text = "```python " + example_text + "```"
#             return_msg += example_text
#
#         if "### Code" in text:
#             example_text = text.split("### Code")[-1]
#             example_text = example_text.split("###")[0]
#             example_text = "```python " + example_text + "```"
#             return_msg += example_text
#
#         return return_msg
#
#     else:
#         return text


# def run_query(llm_chain, question, memory, multiturn, index=None):
#     if index:
#         context = index.similarity_search(question, k=2)
#         logger.info(f"Fetched context: {context}")
#         result = llm_chain.run({"question": question, "context": context})
#         memory.clear()
#     else:
#         result = llm_chain.run(question)
#         if not multiturn:
#             memory.clear()
#     parsed_response = parse_response(result)
#     return parsed_response

async def run_query(llm_chain, question, memory, callback, multiturn, top_p, top_k, max_new_tokens, index):
    print(question)
    if index:
        context = index.similarity_search(question, k=2)
        logger.info(f"Fetched context: {context}")
        result = await llm_chain.arun(question=question, context=context, callbacks=[callback], top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens)
        memory.clear()
    else:
        result = await llm_chain.arun(question=question, callbacks=[callback], top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens)
        if not multiturn:
            memory.clear()
    # parsed_response = parse_response(result)
    return result

