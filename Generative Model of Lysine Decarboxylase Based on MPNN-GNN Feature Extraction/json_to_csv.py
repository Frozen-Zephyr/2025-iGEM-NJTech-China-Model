import json
import pandas as pd

# 读取 JSON 文件
with open("/Users/zephyr/Documents/PycharmProjects/2025igem/DLKcat/DeeplearningApproach/Data/database/Kcat_combination_0918_wildtype_mutant.json", "r") as f:
    data = json.load(f)

# 转成 DataFrame
df = pd.DataFrame(data)

# 导出为 CSV
df.to_csv("data.csv", index=False)
print("已保存为 data.csv")