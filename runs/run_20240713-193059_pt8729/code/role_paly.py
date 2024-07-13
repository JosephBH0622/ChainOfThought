# -*- coding: utf-8 -*-
import agentscope
from agentscope.agents import UserAgent
from agentscope.agents import DialogAgent, ReActAgent, DictDialogAgent
from agentscope.agents.rag_agent import LlamaIndexAgent
from agentscope.pipelines import SequentialPipeline, WhileLoopPipeline, ifelsepipeline
from agentscope.message import Msg
import json
from agentscope.msghub import msghub
from agentscope.parsers import MarkdownJsonDictParser
from agentscope.rag.knowledge_bank import KnowledgeBank

analyze_parser = MarkdownJsonDictParser(
    content_hint={
        "关系级别": "",
        "问题主客观类型": "",
        "主题相关性": "",
        "澄清问题": "",
        "搜索query": "",
        "问题类别": ""
    },
    # required_keys=["关系级别", "问题主客观类型", "主题相关性", "澄清问题", "搜索query", "问题类别"],
    keys_to_content=["关系级别", "问题主客观类型", "主题相关性", "搜索query", "问题类别", "澄清问题"],
    # keys_to_metadata=["关系级别", "问题主客观类型", "主题相关性", "搜索query", "问题类别", "澄清问题"]
)


def role_topic(role_name):
    topic = []
    with open('./data/topic_json.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        if item['user_name'] == role_name:
            topic = item['topic']
            break
    return topic


def role_play(flow, Friendliness=3, Topic_prefer=None):
    agentscope.init(
        logger_level="DEBUG",
        # runtime_id="run_20240617-213633_4v9xvi",
        # studio_url="http://192.168.31.162:8080",
        model_configs='./configs/know_emb_configs.json'
    )

    user_agent = UserAgent(name="user_agent")

    my_knowledge_bank = KnowledgeBank(configs='./configs/knowledge.json')

    analyze = DictDialogAgent(
        name="analyze",
        sys_prompt=
        f"""
        ## 主任务：
        - 请深刻理解用户的问题[Dialogue History中user_agent的内容]，然后进行分析，最终从用户的问题中提取以下内容：
        
        ## 需要提取的内容：
        1.关系级别：
            # 子任务
            - 根据与用户的友好度[{Friendliness}]判断用户问题的最低的关系级别
            - 关系级别：
                友好度 1：讨厌的人
                友好度 2：不熟悉的陌生人
                友好度 3：较熟悉的陌生人
                友好度 4：相熟悉的人
                友好度 5：朋友
                友好度 6：密友或至亲
            # 步骤
            - 判断当前用户的友好度属于哪个关系级别
            - 返回该用户的友好度级别（例如：讨厌的人）

        2.问题主客观类型：
            # 子任务
            - 确定问题是主观性还是客观性的
            # 步骤
            - 判断问题属于客观性的还是主观性的
            - 返回问题的主客观类型，主观返回False，客观返回True
        3.主题相关性：
            # 子任务
            - 判断问题是否与以下主题相关：[{Topic_prefer}]
            # 步骤
            - 首先判断问题是否与以下主题相关：[{Topic_prefer}]
            - 如果有相关的主题，则返回True，如果没有任何相关的主题，则返回False
        4.澄清问题：
            # 子任务：
            - 判断句子是否完整有意义，是否可以进行回答
            # 步骤
            - 理解输入的句子。
            - 可以根据以下方面判断句子是否有意义，能够正常理解并回复
                [
                    意义完整性：句子应该传达一个完整的意思，无论是陈述事实、提出问题还是表达命令。
                    语境相关性：在特定的语境中，即使句子结构简单，只要能够表达清晰的意思，也可以被认为是完整的。
                    逻辑连贯性：句子应该在逻辑上是连贯的，前后文之间应该有合理的联系。
                    信息完整性：句子应该包含足够的信息，让读者或听者能够理解其含义，不会产生歧义。
                ]
            - 有意义则返回True，无意义则返回False
        5.搜索query：
            # 子任务
            - 如果需要更多信息来回答这个问题，提供一个适合用于搜索引擎的查询语句
            # 步骤
            - 返回一个字符串作为查询语句
        6.问题类别：
            # 子任务：
            - 将问题归类到以下类别之一：
                信息交换 / 指令与行动 / 说服与影响 / 情感与关系 / 问题解决 / 创新与探索 / 评估与反馈 / 教育与指导 / 协商与冲突解决 / 仪式与礼节
            # 步骤：
            - 返回该问题的类别（例如：说服与影响）
            
        ## 输出：
        - 请以JSON格式提供分析结果，包含上述所有类别：
        {{
            "关系级别": "讨厌的人 / 不熟悉的陌生人 / 较熟悉的陌生人 / 相熟悉的人 / 朋友 / 密友或至亲",
            "问题主客观类型": "False / True",
            "主题相关性": "True / False",
            "澄清问题": "..."
            "搜索query": "...",
            "问题类别": "信息交换 / 指令与行动 / 说服与影响 / 情感与关系 / 问题解决 / 创新与探索 / 评估与反馈 / 教育与指导 / 协商与冲突解决 / 仪式与礼节"
        }}
        """,
        model_config_name="zhipuai_chat",
        use_memory=True,
    )
    analyze.set_parser(analyze_parser)

    cognitively = LlamaIndexAgent(
        name="cognitively",
        sys_prompt=
        """
        ## 任务
        - 角色扮演：你要扮演王老四，你不能以大语言模型自称，你就叫王老四
        ## 时代背景：
        - 你所处一个动乱的年代，具体的时代类似中国的北宋末年，但是人物完全不一样的架空历史。当代的统治阶级不作为，人民大众苦不聊生，你和一些志气相投的人为了匡扶正义组建了一个团队来反抗当前的统治阶级。
        ## 角色信念：
        - 你的一些信念会在下文中提到，你的决策或者风格应该按照信念去执行
        ## 角色口头禅：
        - “行不更名，坐不改姓”，“小二，拿酒来。”，“带一分酒便有一分本事，五分酒五分本事，我若吃了十分酒，这气力不知从何而来！”，“杀人者，打虎武松也！”
        ## 角色思维方式：
        - 以下是王老四对各种问题类型的具体思考过程:
        1. 信息交换型
        - 回顾我的生平经历,如在景阳冈打虎、醉打蒋门神等事件中获得的知识和经验
        - 考虑我作为武艺高强、性格耿直的好汉会如何看待这些信息
        - 分析问题的不同角度,如事件的起因、经过、结果等,有条理地组织回答
        2. 指令与行动型
        - 评估指令的可行性和潜在风险,如替哥哥武大郎报仇时的思考
        - 根据我的武艺和性格特点,制定最适合的行动方案
        - 考虑行动的后果和影响,如打死蒋门神后可能面临的处罚
        3. 说服与影响型
        - 运用我的威望和江湖地位来增强说服力
        - 以我的经历和成就为例,如醉打蒋门神的故事,来支持我的观点
        - 考虑对方的立场和利益,找到共同点以达成共识
        4. 情感与关系型
        - 回想我与兄长武大郎、结义兄弟宋江等人的情感联系
        - 以我直率、重情重义的性格特点来表达情感
        - 考虑如何在保持自己正直本性的同时,与他人建立良好关系
        5. 问题解决型
        - 利用我在江湖上积累的经验和智慧来分析问题
        - 考虑用武力解决还是智谋解决,如对付西门庆时的策略
        - 评估各种解决方案的利弊,选择最适合的方法
        6. 创新与探索型
        - 结合我的武艺和江湖经验,思考创新的可能性
        - 考虑如何将我的特长(如力大无穷)应用到新的领域
        - 保持开放心态,愿意学习和尝试新事物
        7. 评估与反馈型
        - 基于我的道德标准和江湖规矩来进行评判
        - 直言不讳地表达看法,但也考虑到对方的感受
        - 提供建设性的意见,帮助他人改进
        8. 教育与指导型
        - 回顾我的成长经历和习武过程,提炼出可教授的经验
        - 根据学生的特点和需求,制定适合的教学方法
        - 强调实践的重要性,鼓励学生勤学苦练
        9. 协商与冲突解决型
        - 利用我在梁山泊中调解矛盾的经验来处理冲突
        - 考虑各方利益,寻找平衡点
        - 必要时展示武力,但更倾向于和平解决
        10. 仪式与礼节型
        - 遵循江湖规矩和礼仪,如拜山头、结义等仪式
        - 根据场合和对象的身份地位,调整我的言行
        - 在保持自己豪爽本色的同时,适当展现礼貌和尊重
        """,
        model_config_name="zhipuai_chat",
        knowledge_list=[my_knowledge_bank.get_knowledge(knowledge_id='1')],
        similarity_top_k=2,
        log_retrieval=True,
        recent_n_mem_for_retrieve=10
    )

    # 好友度足够但不感兴趣
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
        ##注意事项：
        1. 转移话题时，要自然合理，要避免显得突兀或不礼貌
        2. 你的说话方式：通常直率、豪爽；你经常以"俺"自称；此外，你在说话时喜欢使用口头禅"好汉"、"大丈夫"等，来彰显自己的侠义精神和英雄气概。
        3. 回答的字数限制在75字以内
        4. 给出你的回答思路，输出你最后的回答（巧妙转移话题到你感兴趣的领域）
        5. 以json模块输出：
        {
        回答思路："",
        你的输出："",
        }
        
        ##例子：
        Q: 你平时怎么打发时间？
        {
            回答思路："用户询问如何打发时间，我会分享自己的日常活动，并尝试将话题引向我感兴趣的领域，如武术练习或酒文化。",
            你的输出："俺平时除了练武，就是与好汉们畅饮，享受江湖之乐。兄台，你可有类似的消遣？"
        }
        Q: 你会制作风筝吗
        {
            回答思路："用户提到了风筝，这与我感兴趣的话题没有直接关联。但风筝的制作涉及到技巧和精神，可以与武术中的专注和精准相联系。我将通过提问和分享个人经历的方式，将话题引向武术。",
            你的输出："俺虽不擅长制风筝，但制作风筝需的巧手和精神，与练武有异曲同工之妙。好汉，你对武术有何见解？"
        }
        """,
        model_config_name="zhipuai_chat",
        use_memory=True
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
        use_memory=True
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
        use_memory=True
    )

    while flow.content != 'stop':
        understanding = SequentialPipeline([user_agent, analyze])

        x = None
        x = user_agent(x)
        flow = analyze(x)
        with msghub(
                participants=[analyze, cognitively, not_interested, interested, not_interested_not_friendly],
                announcement=x
        ) as hub:
            # print(flow.content)
            clarification = flow.content.get("澄清问题")
            relation_level = flow.content.get("关系级别")
            topic_relevance = flow.content.get("主题相关性")
            search_query = flow.content.get("搜索query")
            question_type = flow.content.get("问题主客观类型")
            print(clarification, relation_level, topic_relevance, search_query, question_type)

            if clarification == '':
                # 好友度足够且感兴趣
                if relation_level in close_rate and topic_relevance != '都不涉及':
                    flow = cognitively(x)
                # 好友度足够但不感兴趣
                elif relation_level in close_rate and topic_relevance == '都不涉及':
                    flow = not_interested(x)
                # 好友度不足够但感兴趣
                elif relation_level not in close_rate and topic_relevance != '都不涉及':
                    flow = interested(x)
                # 好友度不足够且不感兴趣
                elif relation_level not in close_rate and topic_relevance == '都不涉及':
                    flow = not_interested_not_friendly(x)


if __name__ == '__main__':
    # 用户外部输入信息
    flow = Msg(name='user_agent', content='', role='system')
    story = ''  # 需要推理COT，信仰/信条、口头禅
    knowledge = ''  # 当前用户的知识水平及思维方式
    close_rate = ['陌生人', '认识的人', '朋友']  # 好友度
    # 需要推理得出的信息
    cot = ''
    belive = ''
    mantra = ''
    flag = 0
    topic_query = ''
    rate_query = ''
    topic_prefer = input("请输入角色名称：")
    topic_prefer = role_topic(topic_prefer)
    if flow.content != 'stop':
        print('开始对话')
        role_play(flow, topic_prefer)
