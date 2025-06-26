from openai import OpenAI
import pandas as pd
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Prepare few-shot samples
few_shot_samples = pd.DataFrame(
    {
        "word": ["defeat", "bat", "sat"],
        "usage": [
            "It was a narrow defeat.",
            "He saw that bat.",
            "The cat sat on the mat.",
        ],
        "labels": ["ambiguous", "ambiguous", "clear"],
        "description": [
            "'defeat' can have a literal interpretation or a figurative one. ",
            "'bat' can be the animal or a verb.",
            "no ambiguity.",
        ],
    }
)

model_inputs = [
    {
        "role": "system",
        "content": "You are a helpful assistant specialized in lexical semantics. Given a word and its usage, classify the word as 'ambiguous' if the word is used in a way that could be interpreted in multiple ways or as 'clear' if it is used in a way that is unambiguous.",
    }
]

for j in range(2):
    model_inputs.append(
        {
            "role": "user",
            "content": f"Word: {few_shot_samples.loc[j, 'word']}\nUsage: {few_shot_samples.loc[j, 'usage']}",
        }
    )
    model_inputs.append(
        {
            "role": "assistant",
            "content": f"Reasoning: {few_shot_samples.loc[j, 'description']}\nAnswer: {few_shot_samples.loc[j, 'labels']}",
        }
    )

# Load data
data = pd.read_json("data/wic.train.json")
data = pd.concat(
    [
        data[["LEMMA", "USAGE_x"]].rename(
            columns={"LEMMA": "lemma", "USAGE_x": "usage"}
        ),
        data[["LEMMA", "USAGE_y"]].rename(
            columns={"LEMMA": "lemma", "USAGE_y": "usage"}
        ),
    ],
    ignore_index=True,
)

try:
    for i, row in data.iterrows():
        print("Processing example", i)
        messages = model_inputs + [
            {
                "role": "user",
                "content": f"Word: {row['lemma']}\nUsage: {row['usage']}",
            },
        ]

        response = (
            client.chat.completions.create(model="gpt-4o-mini", messages=messages)
            .choices[0]
            .message.content
        )

        desc, answer = response.split("\n")
        desc = desc.replace("Reasoning: ", "")
        answer = answer.replace("Answer: ", "")

        data.loc[i, "answer"] = answer
        data.loc[i, "description"] = desc

        print("Word: ", row["lemma"])
        print("Usage: ", row["usage"])
        print("Reasoning: ", desc)
        print("Answer: ", answer)
except KeyboardInterrupt:
    print("interrupted")

data.dropna().to_json("data/wic_chatgpt_annotation.json", orient="records", indent=2)
