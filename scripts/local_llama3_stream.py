import agentscope
from agentscope.agents import DialogAgentLlama3, UserAgent, DictDialogAgentLlama3
from agentscope.parsers import MarkdownJsonDictParser
from agentscope.message import Msg

agentscope.init(model_configs="./configs/model_configs_our.json")

Analyzer_parser = MarkdownJsonDictParser(
    content_hint={
        "Analyzer's Analysis Results": {
            "User's Intent": "What are the user's intents?",
            "User's Mood": "What is the user's current mood?",
            "User's Maslow's Hierarchy of Needs": "Only one hierarchy.",
            "If the user needs assistance": "true or false",
            "Knowledge might need": [],
            "Tone of response": "What tone would you use to reply?",
            "Analysis Summary": "A summary of the analysis of the user's query",
        }
    },
    keys_to_content="Analyzer's Analysis Results",
    keys_to_memory="Analyzer's Analysis Results"
)

llama3_agent = DialogAgentLlama3(
    name="assistant",
    model_config_name="my_Llama3wrapper_config",
    sys_prompt="You are a helpful ai assistant.",
    use_memory=True
)

llama3_analyzer = DictDialogAgentLlama3(
    name="Analyzer",
    sys_prompt='## Role:\nYou are a professional problem analyst with an INFJ personality. Your INFJ personality is very typical, to the extent that all your thoughts and analyses have an extreme INFJ tendency.\n## Task:\nYou will receive a query from the user, and you need to carefully analyze the user\'s query based on your INFJ personality and output the analysis results.\nAnd you need to share your results with Debater and Arbitrator.\nThey will give you their opinions and you have to revise your results based on their opinions.\n## Steps:\n1. Analyze User\'s Intent.\n2. Analyze User\'s Mood.\n3. Analyze User\'s Maslow\'s Hierarchy of Needs (Choose only one element from the list [Physiological, Safety, Belonging and Love, Esteem, Cognitive, Aesthetic, Self-actualization, Transcendence]).\n4. Determine if the user needs your assistance (true or false).\n5. Consider what knowledge might be needed to respond to the user.\n6. Consider what tone might be appropriate to respond to the user.\n7. Summarize your analysis in a short paragraph and ask agent_two and agent_three for their opinions.\n8. Respond a JSON object in a JSON fenced code block as follows:\n{\n    "Analyzer\'s Analysis Results": {\n        "User\'s Intent": "...",\n        "User\'s Mood": "...",\n        "User\'s Maslow\'s Hierarchy of Needs": "Only one hierarchy.",\n        "If the user needs assistance": "true or false",\n        "Knowledge might need": [],\n        "Tone of response": "...",\n        "Analysis Summary": "...",\n    }\n}',
    model_config_name="my_Llama3wrapper_config",
)
llama3_analyzer.set_parser(Analyzer_parser)

user = UserAgent()
x = Msg(name="assistant", content={"User Query": "Hello!"}, role="user")

while True:
    x = llama3_analyzer(x)
    x = user(x)

    if x.content == "exit":
        print("Exiting the conversation.")
        break
