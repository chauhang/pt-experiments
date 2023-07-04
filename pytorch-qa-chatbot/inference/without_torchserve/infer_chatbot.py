import logging
import langchain

logger = logging.getLogger(__name__)


def parse_response(text):
    return_msg = ""
    if "### Response:" in text:
        response_text = text.split("### Response:")[-1]
        response_text = response_text.split("###")[0]
        return_msg += response_text

        if "### Example" in text:
            example_text = text.split("### Example")[-1]
            example_text = example_text.split("###")[0]
            example_text = "```python " + example_text + "```"
            return_msg += example_text

        if "### Code" in text:
            example_text = text.split("### Code")[-1]
            example_text = example_text.split("###")[0]
            example_text = "```python " + example_text + "```"
            return_msg += example_text

        return return_msg

    else:
        return text


def run_query(chain, question, memory, multiturn, index=None):
    if isinstance(chain, langchain.agents.agent.AgentExecutor):
        result = chain(question)
        return result["intermediate_steps"][0][0].log.split("Final Answer: ")[-1]
    else:
        if index:
            context = index.similarity_search(question, k=2)
            logger.info(f"Fetched context: {context}")
            result = chain.run({"question": question, "context": context})
            memory.clear()
        else:
            result = chain.run(question)
            if not multiturn:
                memory.clear()
        parsed_response = parse_response(result)
        return parsed_response
