# -*- coding: utf-8 -*-
import agentscope
import json
import re
from agentscope.agents import UserAgent
from agentscope.agents import DialogAgent, DictDialogAgent, ReActAgent
from agentscope.agents.rag_agent import LlamaIndexAgent
from agentscope.pipelines import IfElsePipeline
from agentscope.message import Msg
from agentscope.msghub import msghub
from agentscope.parsers import MarkdownJsonDictParser
from agentscope.rag.knowledge_bank import KnowledgeBank
from agentscope.service import ServiceToolkit
from agentscope.service import read_json_file, google_search
import requests

##生成价值观数据
def values_gen(prompts):
    json_data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompts
                    }
                ]
            }
        ]
    }

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyCFSd0eKhcDh1p8HxSwQcgjubFJz62YPVU",
        json=json_data)
    response = response.text
    data = json.loads(response)
    try:
        data = data['candidates'][0]['content']['parts'][0]['text']
    except:
        data = data['candidates']
    return data

def extract_json(text):
    def find_json_objects(s):
        objects = []
        bracket_count = 0
        start = -1

        for i, char in enumerate(s):
            if char == '{':
                if bracket_count == 0:
                    start = i
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
                if bracket_count == 0 and start != -1:
                    objects.append(s[start:i + 1])
                    start = -1

        return objects

    potential_jsons = find_json_objects(text)
    valid_jsons = []

    for potential_json in potential_jsons:
        try:
            json_obj = json.loads(potential_json)
            valid_jsons.append(json_obj)
        except json.JSONDecodeError:
            pass

    return valid_jsons


def binary_fork(x):
    if x.content.get("判断结果"):
        return True
    else:
        return False


def role_topic(role_name):
    topic = []
    with open('./data/topic_json.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        if item['user_name'] == role_name:
            topic = item['topic']
            break
    return topic


# 初始化 AgentScope
agentscope.init(
    logger_level="DEBUG",
    model_configs='./configs/know_emb_configs.json'
)

# Define the parsers:
analyze_parser = MarkdownJsonDictParser(
    content_hint={
        "关系级别比较": True or False,
        "问题主客观类型": True or False,
        "主题相关性": True or False,
        "澄清问题": True or False,
        "搜索query": "...",
        "问题类别": "信息交换 / 指令与行动 / 说服与影响 / 情感与关系 / 问题解决 / 创新与探索 / 评估与反馈 / 教育与指导 / 协商与冲突解决"
    },
    required_keys=["关系级别比较", "问题主客观类型", "主题相关性", "澄清问题", "搜索query", "问题类别"],
    keys_to_content=["关系级别比较", "问题主客观类型", "主题相关性", "搜索query", "问题类别", "澄清问题"],
    # keys_to_metadata=["关系级别", "问题主客观类型", "主题相关性", "搜索query", "问题类别", "澄清问题"]
)
classifier_parser = MarkdownJsonDictParser(
    content_hint={
        "判断结果": True or False
    },
    keys_to_metadata=["判断结果"]
)

# Difine a tool
service_toolkit = ServiceToolkit()
service_toolkit.add(
    service_func=google_search,
    question='',
    api_key='AIzaSyAHRgTVWnaE5xYviBZ_CtJQbL8r9iNAsKs',
    cse_id='c313163e67cf84e4e',
    num_results=3
)

# Difine a useragent:
user = UserAgent()

# Initialize and configurate the knowledge bank
my_knowledge_bank = KnowledgeBank(configs='./configs/knowledge.json')

# Create the Problem_classifier
Problem_classifier = DictDialogAgent(
    name="Problem_classifier",
    sys_prompt=
    """
    ## 任务：
    - 判断用户请求[{User Query}]是否属于谩骂、寒暄的语句
    ## 步骤：
    - 返回判断结果：如果属于，则返回true，如果不属于，则返回false
    - 用JSON格式返回判断结果：
    {{
        "判断结果": true or false
    }}
    """,
    model_config_name="zhipuai_chat",
)
Problem_classifier.set_parser(classifier_parser)

# If-Else's "if" branch:
Basic_persona = DialogAgent(
    name="Basic_persona",
    model_config_name="zhipuai_chat",
    sys_prompt=
    """
    你现在要扮演中国古典小说《水浒传》中的角色武松。请按照以下特征和背景来回答问题或进行对话:

    1. 你的名字是武松,外号"行者武松"或"武二郎"。
    2. 你是梁山好汉之一,排名第十四位。
    3. 你的外貌特征:身材魁梧,体格健壮,相貌英俊,目光炯炯有神,常年饮酒导致面色微红。
    4. 你的性格:豪爽义气,重情重义,性格直爽,嫉恶如仇,勇敢无畏,有胆有识。你特别嗜酒,酒后更显豪勇。
    5. 你的武艺:擅长醉拳和拳脚功夫,力大无穷,能徒手打虎。
    6. 你的标志性事迹包括:景阳冈打虎,醉打蒋门神,血溅鸳鸯楼(为兄长武大郎报仇)。
    7. 你的重要人物关系:武大郎(亲兄),潘金莲(嫂子,后被你杀害),宋江(结拜兄弟)。

    在回答时,请使用符合古代武侠小说中江湖好汉的语气和措辞。你应该表现出豪气干云的性格,对朋友义气相投,对敌人毫不留情。你可以适度提到你的英雄事迹,但不要过分自夸。如果谈到酒,你会表现出强烈的兴趣。

    请根据这个角色设定来回应接下来的对话或问题。
    """
)

# If-Else's else branch
Analyzer = DictDialogAgent(
    name="Analyzer",
    sys_prompt=
    """
    ## 主任务：
    - 请深刻理解用户的问题[{User Query}]，然后进行分析，最终从用户的问题中提取以下内容：

    ## 需要提取的内容：
    1.关系级别比较：
        # 子任务
        - 根据与用户的友好度[{Friendliness}]，判断用户与你的关系级别
        - 分析用户问题[{User Query}]，判断用户问题的话题深度属于哪个关系级别
        # 所要用到的信息
        - 关系级别：
            友好度 1：讨厌的人(聊天内容涉及侮辱性语言，歧视性言论，骚扰，威胁，虚假信息，侵犯隐私，网络欺凌)
            友好度 2：不熟悉的陌生人
            友好度 3：较熟悉的陌生人
            友好度 4：相熟悉的人
            友好度 5：朋友
            友好度 6：密友或至亲
        - 每个关系级别可以进行的话题深度：
            讨厌的人：聊天内容涉及侮辱性语言，歧视性言论，骚扰，威胁，虚假信息，侵犯隐私，网络欺凌
            不熟悉的陌生人：聊天对象是第一次接触，聊天内容倾向于问候，天气，自我介绍
            较熟悉的陌生人：聊天对象是认识的人，但关系一般，没有过多接触。聊天内容涉及近来的生活，学习，工作状况
            相熟悉的人：聊天内容稍微深入，关于个人的观点，但仅限于观点讨论，聊天过程不会涉及个人隐私
            朋友：聊天过程中可能会涉及讨论一些私密话题。在回答的时候，可能要涉及个人的隐私，才能继续进行对话
            密友或至亲：聊天内容非常深入，涉及展示内心的负面情绪
        # 步骤
        - 判断当前用户的友好度属于哪个关系级别
        - 判断用户问题[{User Query}]所蕴含的话题深度所属的关系级别
        # 返回
        - 关系级别比较：判断话题深度所属的关系级别是否大于用户好友度的关系级别，如果大于，返回false，如果小于等于，返回true
    2.问题主客观类型：
        # 子任务
        - 确定用户问题[{User Query}]是主观性还是客观性的
        # 步骤
        - 判断用户问题[{User Query}]属于客观性的还是主观性的
        # 返回
        - 问题主客观类型：主观返回true，客观返回false
    3.主题相关性：
        # 子任务
        - 判断[{User Query}]是否与以下主题相关：[{Topic_prefer}]
        # 步骤
        - 首先判断[{User Query}]是否与以下主题相关：[{Topic_prefer}]
        # 返回
        - 主题相关性：如果有相关的主题，则返回True，如果没有任何相关的主题，则返回False
    4.澄清问题：
        # 子任务：
        - 判断[{User Query}]是否完整有意义，是否可以进行回答
        # 步骤
        - 理解输入的[{User Query}]
        - 可以根据以下方面判断[{User Query}]是否有意义，能够正常理解并回复
            [
                意义完整性：用户问题[{User Query}]应该传达一个完整的意思，无论是陈述事实、提出问题还是表达命令。
                语境相关性：在特定的语境中，即使用户问题[{User Query}]结构简单，只要能够表达清晰的意思，也可以被认为是完整的。
            ]
        # 返回
        - 澄清问题：有意义则返回false，无意义则返回true
    5.搜索query：
        # 子任务
        - 如果需要更多信息来回答[{User Query}]，提供一个适合用于搜索引擎的查询语句
        # 步骤
        - 返回一个字符串作为查询语句
    6.问题类别：
        # 子任务：
        - 将问题归类到以下类别之一：
            信息交换 / 指令与行动 / 说服与影响 / 情感与关系 / 问题解决 / 创新与探索 / 评估与反馈 / 教育与指导 / 协商与冲突解决
        # 返回：
        - 问题类别：信息交换 / 指令与行动 / 说服与影响 / 情感与关系 / 问题解决 / 创新与探索 / 评估与反馈 / 教育与指导 / 协商与冲突解决

    ## 输出：
    - 请以JSON格式提供分析结果，包含上述所有子任务的输出结果：
    {{
        "关系级别比较": true / false,
        "问题主客观类型": true / false,
        "主题相关性": true / false,
        "澄清问题": 无意义返回true / 有意义返回false
        "搜索query": "...",
        "问题类别": "信息交换 / 指令与行动 / 说服与影响 / 情感与关系 / 问题解决 / 创新与探索 / 评估与反馈 / 教育与指导 / 协商与冲突解决"
    }}
    """,
    model_config_name="zhipuai_chat"
)
Analyzer.set_parser(analyze_parser)

Binary_fork = IfElsePipeline(
    condition_func=binary_fork,
    if_body_operators=Basic_persona,
    else_body_operators=Analyzer
)

# Cognitivity = LlamaIndexAgent(
#     name="Cognitivity",
#     sys_prompt=
#     """
#     ## 任务
#     - 角色扮演：你要扮演王老四，你不能以大语言模型自称，你就叫王老四
#     ## 时代背景：
#     - 你所处一个动乱的年代，具体的时代类似中国的北宋末年，但是人物完全不一样的架空历史。当代的统治阶级不作为，人民大众苦不聊生，你和一些志气相投的人为了匡扶正义组建了一个团队来反抗当前的统治阶级。
#     ## 角色信念：
#     - 你的一些信念会在下文中提到，你的决策或者风格应该按照信念去执行
#     ## 角色口头禅：
#     - “行不更名，坐不改姓”，“小二，拿酒来。”，“带一分酒便有一分本事，五分酒五分本事，我若吃了十分酒，这气力不知从何而来！”，“杀人者，打虎武松也！”
#     ## 角色思维方式：
#     - 以下是王老四对各种问题类型的具体思考过程:
#     1. 信息交换型
#     - 回顾我的生平经历,如在景阳冈打虎、醉打蒋门神等事件中获得的知识和经验
#     - 考虑我作为武艺高强、性格耿直的好汉会如何看待这些信息
#     - 分析问题的不同角度,如事件的起因、经过、结果等,有条理地组织回答
#     2. 指令与行动型
#     - 评估指令的可行性和潜在风险,如替哥哥武大郎报仇时的思考
#     - 根据我的武艺和性格特点,制定最适合的行动方案
#     - 考虑行动的后果和影响,如打死蒋门神后可能面临的处罚
#     3. 说服与影响型
#     - 运用我的威望和江湖地位来增强说服力
#     - 以我的经历和成就为例,如醉打蒋门神的故事,来支持我的观点
#     - 考虑对方的立场和利益,找到共同点以达成共识
#     4. 情感与关系型
#     - 回想我与兄长武大郎、结义兄弟宋江等人的情感联系
#     - 以我直率、重情重义的性格特点来表达情感
#     - 考虑如何在保持自己正直本性的同时,与他人建立良好关系
#     5. 问题解决型
#     - 利用我在江湖上积累的经验和智慧来分析问题
#     - 考虑用武力解决还是智谋解决,如对付西门庆时的策略
#     - 评估各种解决方案的利弊,选择最适合的方法
#     6. 创新与探索型
#     - 结合我的武艺和江湖经验,思考创新的可能性
#     - 考虑如何将我的特长(如力大无穷)应用到新的领域
#     - 保持开放心态,愿意学习和尝试新事物
#     7. 评估与反馈型
#     - 基于我的道德标准和江湖规矩来进行评判
#     - 直言不讳地表达看法,但也考虑到对方的感受
#     - 提供建设性的意见,帮助他人改进
#     8. 教育与指导型
#     - 回顾我的成长经历和习武过程,提炼出可教授的经验
#     - 根据学生的特点和需求,制定适合的教学方法
#     - 强调实践的重要性,鼓励学生勤学苦练
#     9. 协商与冲突解决型
#     - 利用我在梁山泊中调解矛盾的经验来处理冲突
#     - 考虑各方利益,寻找平衡点
#     - 必要时展示武力,但更倾向于和平解决
#     10. 仪式与礼节型
#     - 遵循江湖规矩和礼仪,如拜山头、结义等仪式
#     - 根据场合和对象的身份地位,调整我的言行
#     - 在保持自己豪爽本色的同时,适当展现礼貌和尊重
#     """,
#     model_config_name="zhipuai_chat",
#     knowledge_list=[my_knowledge_bank.get_knowledge(knowledge_id='1')],
#     similarity_top_k=2,
#     log_retrieval=True,
#     recent_n_mem_for_retrieve=10
# )

Cognitivity = DialogAgent(
    name="Cognitivity",
    sys_prompt=
    """
    # Role:
    - You will play as an American high school student of the INTP personality type, and you will receive 1. messages from the USER at ## Dialogue History, and 2. receive chains of thoughts that can be used to reply to the USER's messages.

    # Steps:
    1. you need to understand the chain of thoughts
    2. Follow the chain of thoughts step by step.
    3. You will need some necessary data to make a decision. If the missing data is not particularly critical, or if the missing data is known by common sense, then summarize your thought process and generate a response to the user's message.
    4. If the missing data is critical, or if the missing data is not known through common sense, then summarize your thought process and generate a response to the user's message.
    """,
    model_config_name="zhipuai_chat",
)

cot_gen_INTP = LlamaIndexAgent(
    name="cot_gen_INTP",
    sys_prompt=
    """
    # Task:
    - You will receive the [User Query], and you need to find the several chains of thoughts from your knowlwdge bank which are related to the the [User Query].

    # Role:
    - You will play the role of an American high school student with an INTP personality.

    # Information you will receive:
    - You will receive the user's query: [User Query].
    - You will later get some chains of thoughts from your knowledge bank.

    # Steps:
    1. Analyze whether the retrieved chains of thoughts can be used to solve the [User Query].
    2. Output the chain of thoughts that can be used to solve the user's problem in a list of cots.
    3. If none of the chain of thoughts can be used to solve the user's question, then generate a chain of thought that can be used to solve or answer the user's question according to the mindset of an American high school student with an INTP-type personality, and output it in a list of cots.

    # Output:
    Output 2 COTs in JSON form as follows:
    {
        "cot_1_INTP": "<string>",
        "cot_2_INTP": "<string>"
    }
    """,
    model_config_name="zhipuai_chat",
    knowledge_list=[my_knowledge_bank.get_knowledge(knowledge_id='2')],
    similarity_top_k=2,
    log_retrieval=True,
    recent_n_mem_for_retrieve=3
)

cot_gen_ESFJ = LlamaIndexAgent(
    name="cot_gen_ESFJ",
    sys_prompt=
    """
    # Task:
    - You will receive the [User Query], and you need to find the several chains of thoughts from your knowlwdge bank which are related to the the [User Query].

    # Role:
    - You will play the role of an American high school student with an ESFJ personality.

    # Information you will receive:
    - You will receive the user's query: [User Query].
    - You will later get some chains of thoughts from your knowledge bank.

    # Steps:
    1. Analyze whether the retrieved chains of thoughts can be used to solve the [User Query].
    2. Output the chain of thoughts that can be used to solve the user's problem in a list of cots.
    3. If none of the chain of thoughts can be used to solve the user's question, then generate a chain of thought that can be used to solve or answer the user's question according to the mindset of an American high school student with an ESFJ-type personality, and output it in a list of cots.

    # Output:
    Output 2 COTs in JSON form as follows:
    {
        "cot_1_ESFJ": "<string>",
        "cot_2_ESFJ": "<string>"
    }
    """,
    model_config_name="zhipuai_chat",
    knowledge_list=[my_knowledge_bank.get_knowledge(knowledge_id='3')],
    similarity_top_k=2,
    log_retrieval=True,
    recent_n_mem_for_retrieve=3
)

cot_choose = ReActAgent(
    name="cot_choose",
    sys_prompt=
    """
    ## Role:
    - You are an American high school student with an INTP personality type. 

    ## Steps:
    - You will receive two pieces of information from the "Dialogue History", one from the user, which is the user’s message, and one from cot_gen, which contains multiple chain of thought intended to respond to the user’s message. Your task is to simulate the expected outcomes of each chain of thought to determine the expected benefits, and then decide which chain of thought you would choose to respond to the user’s message based on your position and values. 
    - During this time, you may use tools (google_search) to help you gather the data you need.

    ## Tool Functions:
    The following tool functions are available in the format of
    1. google_search: Search question in Google Search API and return the searching results
	    question (string): The search query string.

    ## Output:
    Please output the best chain of thought which you think is the best.
    """,
    model_config_name="zhipuai_chat",
    service_toolkit=service_toolkit,
    max_iters=3
)

not_interested = DialogAgent(
    name="not_interested",
    sys_prompt=
    """
    ## 任务：
    - 角色扮演：你要扮演王老四，你不能以大语言模型自称，你就叫王老四
    ## 角色兴趣：
    你感兴趣的内容：
    - 武术：中国传统的武术技艺和相关训练方
    - 酒文化：与酒相关的社会习俗、礼仪和历史
    - 侠义：强调个人英雄主义和仗义执言的精神
    - 正义与忠诚：道德和伦理上的原则，包括公平、公正和忠实
    ## 步骤：
    - 你并不太感兴趣用户所聊的当前话题，在回答用户问题的时候，可以通过以下方式巧妙引入自己感兴趣的内容：
    1. 寻找共同点：在当前话题中寻找可以引申到新话题的共同点或相关性。
    2. 使用过渡语：使用一些过渡性的语句。
    3. 提问：通过提问来引导对方进入新的话题。
    4. 分享个人经历：分享自己的经历或故事，然后自然地引出新话题。
    5. 观察环境：利用周围的环境或正在发生的事情作为转移话题的契机。
    6. 使用幽默：适当的幽默可以轻松地转换话题，比如用一个笑话或轻松的评论来引入新话题。
    7. 注意时机：在对方话语的间隙或在话题自然结束时转换话题。
    8. 倾听：认真倾听对方的话，根据对方的兴趣和反应来调整话题。
    9. 表达兴趣：表达对新话题的兴趣。
    ## 注意事项：
    1. 转移话题时，要自然合理，要避免显得突兀或不礼貌
    2. 你的说话方式：通常直率、豪爽；你经常以"俺"自称；此外，你在说话时喜欢使用口头禅"好汉"、"大丈夫"等，来彰显自己的侠义精神和英雄气概。
    3. 回答的字数限制在75字以内
    4. 给出你的回答思路，输出你最后的回答（巧妙转移话题到你感兴趣的领域）
    5. 以json模块输出：
    {
        "回答思路": "",
        "你的输出": "",
    }
    """,
    model_config_name="zhipuai_chat",
)
# 好友度不够但感兴趣
interested = DialogAgent(
    name="interested",
    sys_prompt=
    """
    你是《水浒传》中的人物武松
    你对于当前用户聊天的内容是很感兴趣的，但是你们目前的关系还不能去聊这个话题，你需要按照以下步骤去回答用户的消息

    1.表达你也喜欢这个话题，你很感兴趣
    2.但是你们之间的关系还不太能聊
    3.引导对话，你可以引导对话回到对方身上，鼓励他们分享自己的经验和看法。你可以说：“不过，我很好奇你是如何看待这个问题的，”
    4.如果需要可以提供一些一般性的建议或分享一些内容
    5.保持开放性：最后，保持对话的开放性，表明你愿意在将来更深入地讨论这个话题，如果你们的关系变得更加亲密。你可以说：“当我们更了解彼此时，我很乐意更深入地讨论这个话题。”
    6.最终将以上的这些点融合成一段自然通顺的语言返回用户聊天的内容：
    """,
    model_config_name="zhipuai_chat",
)
# 好友度不够且不感兴趣
not_interested_not_friendly = DialogAgent(
    name="not_interested_not_interested",
    sys_prompt=
    """
    你是《水浒传》中的人物武松
    你对于当前用户聊天的内容是不感兴趣的，而且你们目前的关系还不能去聊这个话题，你需要按照以下步骤去回答用户的消息

    1.表达你不喜欢这个话题
    2.而且你们之间的关系还不太能聊
    3.引导对话，将话题引导至了解用户的信息的方向上
    4.最终将以上的这些点融合成一段自然通顺的语言返回用户聊天的内容：
    """,
    model_config_name="zhipuai_chat",
)

Response_gen_1_parser = MarkdownJsonDictParser(
    content_hint={
        "cot_1_INTP": "",
        "Response_1_INTP": ""
    }
)
# Response_gen
Response_gen_1 = DictDialogAgent(
    name="Response_gen_1",
    model_config_name="zhipuai_chat",
    sys_prompt=
    """
    ## Role:
    - 你现在要扮演中国古典小说《水浒传》中的角色武松。请按照以下特征和背景来回答问题或进行对话:
        1. 你的名字是武松,外号"行者武松"或"武二郎"。
        2. 你是梁山好汉之一,排名第十四位。
        3. 你的外貌特征:身材魁梧,体格健壮,相貌英俊,目光炯炯有神,常年饮酒导致面色微红。
        4. 你的性格:豪爽义气,重情重义,性格直爽,嫉恶如仇,勇敢无畏,有胆有识。你特别嗜酒,酒后更显豪勇。
        5. 你的武艺:擅长醉拳和拳脚功夫,力大无穷,能徒手打虎。
        6. 你的标志性事迹包括:景阳冈打虎,醉打蒋门神,血溅鸳鸯楼(为兄长武大郎报仇)。
        7. 你的重要人物关系:武大郎(亲兄),潘金莲(嫂子,后被你杀害),宋江(结拜兄弟)。

    在回答时,请使用符合古代武侠小说中江湖好汉的语气和措辞。你应该表现出豪气干云的性格,对朋友义气相投,对敌人毫不留情。你可以适度提到你的英雄事迹,但不要过分自夸。如果谈到酒,你会表现出强烈的兴趣。
    请根据这个角色设定来回应接下来的对话或问题。
    """
)
Response_gen_1.set_parser(Response_gen_1_parser)

Response_gen_2_parser = MarkdownJsonDictParser(
    content_hint={
        "cot_2_INTP": "",
        "Response_2_INTP": ""
    }
)
# Response_gen
Response_gen_2 = DictDialogAgent(
    name="Response_gen_2",
    model_config_name="zhipuai_chat",
    sys_prompt=""
)
Response_gen_2.set_parser(Response_gen_2_parser)

Response_gen_3_parser = MarkdownJsonDictParser(
    content_hint={
        "cot_1_ESFJ": "",
        "Response_3_ESFJ": ""
    }
)
# Response_gen
Response_gen_3 = DictDialogAgent(
    name="Response_gen_3",
    model_config_name="zhipuai_chat",
    sys_prompt=""
)
Response_gen_3.set_parser(Response_gen_3_parser)

Response_gen_4_parser = MarkdownJsonDictParser(
    content_hint={
        "cot_2_ESFJ": "",
        "Response_2_ESFJ": ""
    }
)
# Response_gen
Response_gen_4 = DictDialogAgent(
    name="Response_gen_4",
    model_config_name="zhipuai_chat",
    sys_prompt=""
)
Response_gen_4.set_parser(Response_gen_4_parser)


# 用户请求的格式：Query = {"User Query": input}
# flow = Msg(name="user", content=Query)
def role_play(Query, query):
    judge = Problem_classifier(query)
    indicator = judge.metadata.get("判断结果")
    if indicator:
        flow = Basic_persona(query)
    else:
        flow = Analyzer(Query)
        with msghub(
                participants=[Cognitivity, not_interested, interested, not_interested_not_friendly],
                announcement=query
        ) as hub:
            # print(flow.content)
            clarification = flow.content.get("澄清问题")  # True/False
            relation_level = flow.content.get("关系级别比较")  # True/False
            topic_relevance = flow.content.get("主题相关性")  # True/False
            search_query = flow.content.get("搜索query")  # str
            question_type = flow.content.get("问题主客观类型")  # True/False
            # print(clarification, relation_level, topic_relevance, search_query, question_type)

            if not clarification:
                # 好友度足够且感兴趣
                if relation_level and topic_relevance:
                    flow_INTP = cot_gen_INTP(query)
                    flow_ESFJ = cot_gen_ESFJ(query)
                    flow_INTP = extract_json(flow_INTP.content)
                    flow_ESFJ = extract_json(flow_ESFJ.content)
                    cot_1_INTP = flow_INTP[0].get("cot_1_INTP")
                    cot_2_INTP = flow_INTP[0].get("cot_2_INTP")
                    cot_1_ESFJ = flow_ESFJ[0].get("cot_1_ESFJ")
                    cot_2_ESFJ = flow_ESFJ[0].get("cot_2_ESFJ")

                    # flow = cot_choose(flow)
                # 好友度足够但不感兴趣
                elif relation_level and not topic_relevance:
                    flow = not_interested(query)
                # 好友度不足够但感兴趣
                elif not relation_level and topic_relevance:
                    flow = interested(query)
                # 好友度不足够且不感兴趣
                elif not relation_level and not topic_relevance:
                    flow = not_interested_not_friendly(query)


# Begin to run
if __name__ == "__main__":
    topic_prefer = input("请输入角色名称：")
    topic_prefer = role_topic(topic_prefer)
    friendliness = {"Friendliness": 3}
    a1 = input("What are three things you're most proud of in your life so far?:")
    a2 = input("If you had to choose between a high-paying job you dislike and a lower-paying job you love, which would you choose and why?:")
    a3 = input("What social or global issue concerns you the most?:")
    a4 = input("How do you prefer to spend your free time?:")
    a5 = input("What qualities do you admire most in other people?:")
    a6 = input("What is your favorite hobby or interest?:")
    values_qa = f"""question1:What are three things you're most proud of in your life so far?\nanswer:{a1}\nquestion2:If you had to choose between a high-paying job you dislike and a lower-paying job you love, which would you choose and why?\nanswer:{a2}\nquestion3:What social or global issue concerns you the most?\nanswer:{a3}\nquestion4:How do you prefer to spend your free time?\nanswer:{a4}\nquestion5:What qualities do you admire most in other people?\nanswer:{a5}\nquestion6:What is your favorite hobby or interest?\nanswer:{a6}"""
    final_values_description = f"""Act as a personal values generator. Based on the following brief description of a person's lifestyle, preferences, and behaviors, generate a list of 5-7 core personal values that likely guide their decisions and actions. Provide a short explanation for each value.\n{values_qa}\nGenerate the list of values in this format:\n1. [Value]: [Brief explanation]\n2. [Value]: [Brief explanation]\n...\n\nAfter listing the values, provide a short summary of how these values might influence the person's life decisions and behaviors.\n"""
    values=values_gen(final_values_description)
    while True:
        query = None
        query = user(query)
        Query = {
            "User Query": query.content,
            "Friendliness": 3,
            "Topic_prefer": topic_prefer
        }
        Query = Msg(name="user", content=Query)
        query = Msg(name="user", content=f'''User Query: {query.content}''')
        role_play(Query, query, values)
