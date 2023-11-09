import json
from pathlib import Path
# Opening JSON file
def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

random_asocial = load_json(Path("llm_log/random_asocial_04_01_2023_14:28:53/evaluation_log.json"))
random_boxes = load_json(Path("llm_log/random_boxes_04_01_2023_14:32:17/evaluation_log.json"))

ada_asocial = load_json(Path("llm_log/ada_asocial_3_04_01_2023_14:53:16/evaluation_log.json"))
ada_boxes = load_json(Path("llm_log/ada_3st_boxes_04_01_2023_18:55:38/evaluation_log.json")) # no caretaker
ada_boxes_c = load_json(Path("llm_log/ada_3st_boxes_caretaker_04_01_2023_20:18:18/evaluation_log.json"))  # caretaker

davinci_asocial = load_json(Path("llm_log/davinci_asocial_3st_04_01_2023_21:27:23/evaluation_log.json"))
davinci_boxes = load_json(Path("llm_log/davinci_3st_boxes_04_01_2023_20:37:28/evaluation_log.json"))
davinci_boxes_c = load_json(Path("llm_log/davinci_3st_boxes_caretaker_04_01_2023_21:17:44/evaluation_log.json"))

bloom_560_asocial = load_json(Path("llm_log/bloom_560m_asocial_3st_04_01_2023_14:59:44/evaluation_log.json"))
bloom_560_boxes = load_json(Path("llm_log/bloom_560_3st_boxes_04_01_2023_20:14:13/evaluation_log.json"))  # no caretaker
bloom_560_boxes_c = load_json(Path("llm_log/bloom_560_3st_boxes_caretaker_04_01_2023_20:05:08/evaluation_log.json")) #  caretaker


data = [
    random_asocial,
    random_boxes,

    ada_asocial,
    # ada_boxes,
    ada_boxes_c,

    davinci_asocial,
    # davinci_boxes,
    davinci_boxes_c,

    bloom_560_asocial,
    # bloom_560_boxes,
    bloom_560_boxes_c,

]

for d in data:
    print(f'Model: {d["model"]} Env: {d["env_name"]} {"hist" if d["feed_full_ep"] else ""} ---> {d["mean_success_rate"]} ({len(d["success_rates"])})')

