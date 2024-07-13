import json
import configs as cf

with open(cf.teresa_json_pth, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 定义一个字典来存储具有相同 Maslow's Need Level 值的事件
events_by_need_level = {}

# 遍历 Story Outline
for story_key, story_value in data.items():
    for event_key, event_value in story_value.items():
        need_level = event_value.get("Maslow's Need Level:", "")

        # 如果这个 need_level 还没有在字典中创建对应的键值对，就创建一个空列表
        if need_level not in events_by_need_level:
            events_by_need_level[need_level] = []

        # 将当前事件添加到相应的 need_level 键的列表中
        events_by_need_level[need_level].append({
            "Story": story_key,
            "Event": event_key,
            "Description": event_value.get("Event's Description:", ""),
            "Learnings": event_value.get("Teresa's Learnings:", ""),
            "Actions": event_value.get("Teresa's Actions:", ""),
            "Emotions": event_value.get("Teresa's Emotions:", ""),
        })

# 将具有相同 Maslow's Need Level 的事件分别存储到不同的 JSON 文件中
for need_level, events in events_by_need_level.items():
    filename = cf.maslow_reallocate + f'events_level_{need_level}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=4, ensure_ascii=False)

    print(f"Events with Maslow's Need Level {need_level} saved to {filename}")
