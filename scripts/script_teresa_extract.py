from docx import Document
import json
import re


def extract_text_from_docx(doc_path):
    doc = Document(doc_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def parse_text_to_json(text):
    story_outlines = re.split(r'\n*Story Outline:', text)[1:]
    result = {}

    for i, outline in enumerate(story_outlines, 1):
        story_outline_key = f"Story Outline {i}"
        events = re.split(r'\n*Event (\d+):', outline)[1:]
        events_dict = {}

        for j in range(0, len(events), 2):
            event_key = f"Event {events[j].strip()}"
            event_content = events[j + 1]
            event_sections = re.split(r'(\w+\'s [\w\s]+:)', event_content)[1:]
            # print(event_sections)

            event_dict = {}
            current_key = None
            current_value = []

            for k in range(0, len(event_sections), 2):
                section_key = event_sections[k].strip()
                section_value = event_sections[k + 1].strip() if k + 1 < len(event_sections) else ""
                event_dict[section_key] = section_value

            events_dict[event_key] = event_dict

        result[story_outline_key] = events_dict

    return result


def convert_docx_to_json(doc_path, json_path):
    text = extract_text_from_docx(doc_path)
    json_data = parse_text_to_json(text)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)


# 使用示例
doc_path = '../data/特蕾莎修女数据.docx'
json_path = '../data/Teresa_data.json'
convert_docx_to_json(doc_path, json_path)
print(f"JSON数据已保存到: {json_path}")
