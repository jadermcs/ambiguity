from openai import OpenAI
from argparse import ArgumentParser
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
data = data.reset_index()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--output", type=str, default="data/wic_chatgpt_annotation.train.jsonl"
    )
    parser.add_argument("--start_index", type=str, default="0")
    args = parser.parse_args()

    skipped = []

    try:
        start_index = args.start_index.split(",")
        if len(start_index) == 1:
            iterator = data.iloc[int(start_index[0]) :]
        else:
            start_index = [int(i) for i in start_index]
            iterator = data.iloc[start_index]
        for i, row in iterator.iterrows():
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

            try:
                desc, answer = response.split("\n")
                desc = desc.replace("Reasoning: ", "")
                answer = answer.replace("Answer: ", "")

                data.loc[i, "answer"] = answer
                data.loc[i, "description"] = desc

                print("Word: ", row["lemma"])
                print("Usage: ", row["usage"])
                print("Reasoning: ", desc)
                print("Answer: ", answer)
                example = pd.DataFrame(
                    {
                        "lemma": row["lemma"],
                        "usage": row["usage"],
                        "description": desc,
                        "answer": answer,
                    },
                    index=[i],
                ).reset_index()
                example.to_json(args.output, orient="records", lines=True, mode="a")
            except ValueError as e:
                print(e)
                print("Invalid response format, skipping example", i)
                skipped.append(i)

    except KeyboardInterrupt:
        print("interrupted")
    # append new data to existing file
    print("Skipped examples:", skipped)


if __name__ == "__main__":
    main()
