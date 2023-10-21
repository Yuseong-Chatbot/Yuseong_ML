import pandas as pd
df = pd.read_csv("data1.csv")
df.fillna("")

text_col = []
for _, row in df.iterrows():
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instruction = str(row["instruction"])
    response = str(row["output"])
    
    text = prompt + "### Instruction:\n" + instruction + "\n### Response:\n" + response
    
    text_col.append(text)

df.loc[:, "text"] = text_col
df.to_csv('train.csv', index=False)

# import csv
# import json

# csv_path = 'data1.csv'
# with open(csv_path, 'r') as f:
#     reader = csv.reader(f)
#     next(reader)

#     data = []
#     for line in reader:
#         d = {
#             'instruction': str(line[0]),
#             'input': str(line[1]),
#             'output': str(line[2])
#         }
#         data.append(d)

# json_str = json.dumps(data, ensure_ascii=False, indent=2)
# with open('train.json', 'w', encoding='utf-8') as f:
#     f.write(json_str)