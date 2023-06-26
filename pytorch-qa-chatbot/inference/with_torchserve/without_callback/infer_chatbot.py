import logging

logger = logging.getLogger(__name__)

def run_query(llm_chain, question, memory, multiturn, index, top_p, top_k, max_new_tokens):
    if index:
        context = index.similarity_search(question, k=2)
        logger.info(f"Fetched context: {context}")
        llm_chain.run(question=question, context=context, top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens)
        memory.clear()
    else:
        llm_chain.run(question=question, top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens)
        if not multiturn:
            memory.clear()
    return