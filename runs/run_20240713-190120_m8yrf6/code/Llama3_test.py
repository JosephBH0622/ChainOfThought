import agentscope
from agentscope.agents import DialogAgent, UserAgent
from agentscope.message import Msg

agentscope.init(model_configs="./configs/know_emb_configs.json")

chat = DialogAgent(
    name="chat",
    sys_prompt="You are a helpful assistant.",
    model_config_name="zhipuai_chat",
    use_memory=True
)

user = UserAgent()

while True:
    flow = None
    flow = user(flow)
    flow = chat(flow)