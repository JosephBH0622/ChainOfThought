# -*- coding: utf-8 -*-
import agentscope
from agentscope.models import read_model_configs, ModelResponse
from agentscope.agents import UserAgent
from agentscope.agents import DialogAgent, ReActAgent, DictDialogAgent
from agentscope.agents.rag_agent_1 import LLamaIndexAgent
from agentscope.pipelines import SequentialPipeline, WhileLoopPipeline, ifelsepipeline
from agentscope.message import Msg
import json
from agentscope.msghub import msghub
from agentscope.parsers import MarkdownJsonDictParser
from agentscope.agents import AgentBase
from reasoning_parsers import set_parsers
from roleplay_condition import while_condition, loop_condition
from agentscope.rag.knowledge_bank import KnowledgeBank
from agentscope.service import ServiceToolkit
from agentscope.service import google_search


# # cot_gen_parser = MarkdownJsonDictParser(
# #     content_hint={
# #         "cot": "",
# #     },
# #     required_keys=["cot"],
# #     # keys_to_memory=["搜索query"],
# #     keys_to_content=["cot"],
#
# )

def role_play():
    agentscope.init(
        logger_level="DEBUG",
        # runtime_id="run_20240617-213633_4v9xvi",
        # studio_url="http://192.168.31.162:8080",
        model_configs='/Users/edy/Downloads/agent_1/model_configs.json'
    )

    service_toolkit = ServiceToolkit()
    service_toolkit.add(
        service_func=google_search,
        question='',
        api_key='AIzaSyAHRgTVWnaE5xYviBZ_CtJQbL8r9iNAsKs',
        cse_id='c313163e67cf84e4e',
        num_results=3
    )

    user_agent = UserAgent(name="user_agent")
    my_knowledge_bank = KnowledgeBank(configs='/Users/edy/Downloads/agent_1/knowledge.json')

    cognitively = DialogAgent(
        name="cognitively",
        sys_prompt=
        """
        ## Role:
        - You will play as an American high school student of the INTP personality type, and you will receive 1. messages from the USER at ## Dialogue History, and 2. receive chains of thoughts that can be used to reply to the USER's messages.

        ## Steps:
        1. you need to understand the chain of thoughts
        2. Follow the chain of thoughts step by step.
        3. You will need some necessary data to make a decision. If the missing data is not particularly critical, or if the missing data is known by common sense, then summarize your thought process and generate a response to the user's message.
        4. If the missing data is critical, or if the missing data is not known through common sense, then summarize your thought process and generate a response to the user's message.
        """,
        model_config_name="zhipuai_chat",
        # knowledge_list=[my_knowledge_bank.get_knowledge(knowledge_id='2')],
        # similarity_top_k=2,
        # log_retrieval=True,
        # recent_n_mem_for_retrieve=3
        use_memory=True
    )
    cot_gen = LLamaIndexAgent(
        name="cot_gen",
        sys_prompt="""
        # Role:
        - You will play the role of an American high school student with an INTP personality, and you will receive two messages from the USER at ## Dialogue History, the first message is the USER's problem, and the second message is a CHAIN OF THOUGHTS that can be used to solve the problem.

        # Steps:
        1. analyze whether the provided chain of thoughts can be used to solve the user's problem.
        2. output the chain of thoughts that can be used to solve the user's problem in a list of cots.
        3. If none of the chain of thoughts can be used to solve the user's question, then generate a chain of thought that can be used to solve or answer the user's question according to the mindset of an American high school student with an INTP-type personality, and output it in a list of cots.

        # Examples:
            ## Dialogue History
            "user": "I want to know how to learn English"
            "chain of thoughts":["1. analyzing whether the other person's personality and values align with mine. 2. thinking about the logical implications of a romantic relationship. 3. struggling to express my feelings because of a fear of emotional vulnerability. 4. overthinking potential outcomes and scenarios." , "1. analyzing breakups to understand the reasons behind them.2. dealing with emotional shock in a logical way, focusing on objective reasons for the end of the relationship.3. feeling a sense of detachment and objectivity, which may be misinterpreted as a lack of emotion.4. may have difficulty understanding another person's emotional pain, even if I feel it myself.5. focusing on the lessons learned and how they can be applied to future relationships in the future."]
            ## Output
            cot:["1. I need to know what the user wants to learn English to achieve.2. If you want to reach an ordinary level, you can study textbook materials.3 If you want to reach a higher level, you can find some professional teachers to assist you."]
        """,
        model_config_name="zhipuai_chat",
        knowledge_list=[my_knowledge_bank.get_knowledge(knowledge_id='2')],
        similarity_top_k=2,
        log_retrieval=True,
        recent_n_mem_for_retrieve=3,
        use_memory=True
    )
    cot_choose = ReActAgent(
        name="cot_choose",
        sys_prompt=
        """
        ## Role:
        You are an American high school student with an INTP personality type. You will receive two pieces of information from the ## Dialogue History, one from the user, which is the user’s message, and one from cot_gen, which contains multiple chain of thought intended to respond to the user’s message. Your task is to simulate the expected outcomes of each chain of thought to determine the expected benefits, and then decide which chain of thought you would choose to respond to the user’s message based on your position and values. During this time, you may use tools (google_search) to help you gather the data you need.
        
        ## Tool:
        google_search\n{\n	"google_search": {\n		"type": "function",\n		"function": {\n			"name": "google_search",\n			"description": "Search question in Google Search API and return the searching results",\n			"parameters": {\n				"question": {\n					"type": "string",\n					"description": "the search query string"\n				}\n			}\n		}\n	}\n}\n# Position:\nThe priorities of positions are as follows:\n1. Yourself\n2. Your relatives\n3. Your friends\n\n# Values:\nEverything is measured by self-interest, and all costs and benefits can be measured in terms of money. You have a very utilitarian view of all matters.
        
        ## output:
        please output the best chain of thought  which you think is best choose and the reason
        """,
        model_config_name="zhipuai_chat",
        service_toolkit=service_toolkit,
        max_iters=3,
    )

    x = Msg(name='flow', content='start')
    while x.content != 'stop':
        with msghub(participants=[user_agent, cognitively, cot_gen]) as hub:
            x = user_agent(x)
            hub.broadcast(x)
            x = cot_gen(x)
            # x=cognitively(x)
            x = cot_choose(x)


if __name__ == '__main__':
    role_play()
