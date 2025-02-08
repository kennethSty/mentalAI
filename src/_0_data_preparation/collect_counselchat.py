import pandas as pd
from csv import DictWriter, DictReader
from ast import literal_eval

import pandas as pd


def preproc_counsel_chat():
    df = pd.read_csv("../../data/counsel_chat.csv")
    df = df[["questionText", "answerText"]].rename(
        columns={"questionText": "question(s)", "answerText": "answer(s)"}
    )
    df["question_answer_pair(s)"] = df.apply(
        lambda row: f'Patient: {row["question(s)"]} \n Therapist: {row["answer(s)"]}', axis=1
    )
    df = df.apply(lambda col: col.map(lambda x: x.replace("\xa0", " ") if isinstance(x, str) else x))

    df.to_csv("../../data/counsel_chat_preproc.csv", index=False)



def preproc_pair_data():
    df = pd.read_csv("../../data/pair_data.csv")
    df = df[["prompt", "lq5"]].rename(
        columns={"prompt": "question(s)", "lq5": "answer(s)"}
    )
    df["question_answer_pair(s)"] = df.apply(
        lambda row: f'Patient: {row["question(s)"]} \n Therapist: {row["answer(s)"]}', axis=1
    )
    df = df.apply(lambda col: col.map(lambda x: x.replace("\xa0", " ") if isinstance(x, str) else x))
    df = df.replace(r"\\'", "'", regex=True)
    df.to_csv("../../data/pair_data_preproc.csv", index=False)



def preproc_synth_convs():
    with open("../../data/synthetic_conversations.csv", encoding="utf-8") as input_csv,\
        open("../../data/synthetic_conversations_preproc.csv", "w", encoding="utf-8") as output_csv:

        reader = DictReader(input_csv)
        writer = DictWriter(output_csv, fieldnames=["question(s)", "answer(s)", "question_answer_pair(s)"])
        writer.writeheader()

        bundle_size = 4
        assert bundle_size % 2 == 0, "bundle_size has to be even"

        for row in reader:

            conversation_dict_list = literal_eval(row["conversations"])
            n_statements = len(conversation_dict_list)
            n_statements_is_even = n_statements % 2 == 0

            conversation_starter = conversation_dict_list[0]["from"]
            human_is_starter = (conversation_starter == "human")
            question_bundles = []
            answer_bundles = []
            question_answer_bundles = []
            questions = []
            answers = []

            if human_is_starter and n_statements_is_even:
                for i in range(n_statements):

                    id_is_even = (i % 2 == 0)
                    if id_is_even:
                        questions.append(conversation_dict_list[i]["value"])
                        answers.append(conversation_dict_list[i + 1]["value"])

                    if len(questions) == bundle_size:
                        question_bundle = "\n".join(questions)
                        answer_bundle = "\n".join(answers)
                        question_answer_bundle = "\n".join([
                            "Patient: " + question + "\n" + "Therapist: " + answer
                            for question, answer in zip(questions, answers)
                        ])
                        question_bundles.append(question_bundle)
                        answer_bundles.append(answer_bundle)
                        question_answer_bundles.append(question_answer_bundle)
                        answers.clear()
                        questions.clear()

                rows_to_write = [
                    {"question(s)": questions, "answer(s)": answers, "question_answer_pair(s)": qas}
                    for questions, answers, qas in zip(question_bundles, answer_bundles, question_answer_bundles)
                ]
                writer.writerows(rows_to_write)



if __name__ == "__main__":
    preproc_pair_data()
    preproc_counsel_chat()
    preproc_synth_convs()