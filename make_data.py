import json
import random
random.seed(42)

def strip_split(text, min_len=40):
    strips = text.split('. ')
    i = 0
    while i < len(strips):
        if len(strips[i]) < min_len:
            # 如果是最后一个元素，无法与后面的元素合并，只能与前面的合并
            if i == len(strips) - 1:
                strips[i - 1] += '. ' + strips[i]
                strips.pop(i)
                i -= 1
            else:
                # 比较当前元素和下一个元素的长度，选择较短的进行合并
                if len(strips[i]) <= len(strips[i + 1]):
                    strips[i] += '. ' + strips[i + 1]
                    strips.pop(i + 1)
                else:
                    strips[i + 1] = strips[i] + '. ' + strips[i + 1]
                    strips.pop(i)
                    i -= 1
        i += 1

    return strips


file_path = '/root/autodl-tmp/triviaqa_test_w_gs.jsonl'
dataset = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        for data in json_obj['ctxs']:
            strips = strip_split(data['text'])
            text = data['title'] + '\n' + data['text']
            if len(strips) < 3:
                continue
            mask_sen = random.choice(strips)
            text = text.replace(mask_sen, '[Mask]')
            dataset.append({
                "instruction": "Your Task is to GUESS the complete sentence to fill in the [Mask] section of the Wiki Document.",
                "input": text,
                "output": mask_sen
            })
    
random.shuffle(dataset)
print(len(dataset))
# 165052

save_path = "/root/autodl-tmp/mask_dataset.json"

with open(save_path, 'w') as file:
    json.dump(dataset, file)

