import json


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


# 示例使用
text = """
这里有一些文本和一个JSON: {"name": "John", "age": 30}
还有另一个JSON: {"city": "New York", "country": "USA"}
这不是有效的JSON: {invalid: json}
嵌套的JSON: {"outer": {"inner": "value"}}
"""

extracted_jsons = extract_json(text)
print(extracted_jsons)