import agentscope
import multiprocessing
from agentscope.message import Msg
from agentscope.rag.knowledge import Knowledge
from agentscope.rag.knowledge_bank import KnowledgeBank, LlamaIndexKnowledge

from typing import Any, Optional
from loguru import logger
from agentscope.models import ModelWrapperBase
from agentscope.agents import LlamaIndexAgent, UserAgent, DialogAgent

agentscope.init(
    model_configs="./configs/Gemini_model_config.json",
    logger_level="DEBUG",
    # runtime_id="run_20240618-161651_vbl5i1",
    studio_url="http://127.0.0.1:8080",
)

# configs = {}
#
# our_knowledge_bank = KnowledgeBank(configs=configs)
#
# our_knowledge_bank.add_data_as_knowledge(
#     knowledge_id="random_knowledge",
#     emb_model_name="Gemini_emb",
#     data_dirs_and_types={
#         "./data": [".xml"],
#     }
# )
#
# our_knowledge = our_knowledge_bank.get_knowledge(knowledge_id="random_knowledge")
#
# Rag_Agent = LlamaIndexAgent(
#     name="Rag_Agent",
#     sys_prompt="Acoording to the user's query, get things the user needs.",
#     model_config_name="Gemini_rag",
#     knowledge_list=[our_knowledge],
#     similarity_top_k=2,
#     log_retrieval=True,
#     recent_n_mem_for_retrieve=1,
# )

agent1 = DialogAgent(
    model_config_name="zhipuai_chat",
    sys_prompt="You are a helpful assistant.",
    name="assistant"
)

agent2 = DialogAgent(
    model_config_name="zhipuai_chat",
    sys_prompt="You are a helpful assistant.",
    name="assistant"
)

user = UserAgent()

flow = None
flow = user(flow)
flow = agent1(flow)
flow = agent2(flow)
# msg1 = Msg(name="bot", content="Help me get some ")
# agent3 = DialogAgent(
#     model_config_name="zhipuai_chat",
#     sys_prompt="You are a helpful assistant.",
#     name="assistant1"
# ).to_dist()


# x = None
# y = None
# # x = user(x)
#
# if __name__ == '__main__':
#     # multiprocessing.freeze_support()
#     while x is None or x.content == "exit":
#         x = agent1(x)
#         y = agent2(y)
#         x = agent1(x)
#         y = agent2(y)
