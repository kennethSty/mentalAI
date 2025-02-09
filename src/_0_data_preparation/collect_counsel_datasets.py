import pandas as pd
from csv import DictWriter, DictReader
from ast import literal_eval
import re

def preproc_counsel_chat():
    df = pd.read_csv("../../data/00_raw/counsel_chat.csv")
    df = df[["questionText", "answerText"]].rename(
        columns={"questionText": "question(s)", "answerText": "answer(s)"}
    )
    df["question_answer_pair(s)"] = df.apply(
        lambda row: f'<Patient>: {row["question(s)"]} \n <Therapist>: {row["answer(s)"]}', axis=1
    )
    df = df.apply(lambda col: col.map(lambda x: x.replace("\xa0", " ") if isinstance(x, str) else x))

    df.to_csv("../../data/01_preprocessed/counsel_chat_preproc.csv", index=False)

def preproc_mental_faq():
    with open("../../data/00_raw/mental_faq.csv", encoding="utf-8") as input_csv, \
            open("../../data/01_preprocessed/mental_faq_preproc.csv", "w", encoding="utf-8") as output_csv:

        reader = DictReader(input_csv)
        writer = DictWriter(output_csv, fieldnames=["question(s)", "answer(s)", "question_answer_pair(s)"])
        writer.writeheader()
        batch_size = 50
        qa_pairs = []
        questions = []
        answers = []

        for row in reader:
            qa_pair = row["text"].replace("<HUMAN>", "<Patient>")\
                .replace("<ASSISTANT>", "<Therapist>")

            question_match = re.search(r"<Patient>:\s*(.*?)\s*<Therapist>", qa_pair, re.DOTALL)
            if question_match:
                question = question_match.group(1)
                answer_match = re.search(r"<Therapist>:\s*(.*)", qa_pair, re.DOTALL)
                if answer_match:
                    answer = answer_match.group(1)
                    # Only collect if a full question and answer was detected
                    qa_pairs.append(qa_pair)
                    questions.append(question)
                    answers.append(answer)

            if len(qa_pairs) == batch_size:
                rows_to_write = [
                    {"question(s)": questions, "answer(s)": answers, "question_answer_pair(s)": qas}
                    for questions, answers, qas in zip(questions, answers, qa_pairs)
                ]
                writer.writerows(rows_to_write)
                questions.clear()
                answers.clear()
                qa_pairs.clear()

        if qa_pairs:
            rows_to_write = [
                {"question(s)": questions, "answer(s)": answers, "question_answer_pair(s)": qas}
                for questions, answers, qas in zip(questions, answers, qa_pairs)
            ]
            writer.writerows(rows_to_write)
            questions.clear()
            answers.clear()
            qa_pairs.clear()


def preproc_pair_data():
    df = pd.read_csv("../../data/00_raw/pair_data.csv")
    df = df[["prompt", "lq5"]].rename(
        columns={"prompt": "question(s)", "lq5": "answer(s)"}
    )
    df["question_answer_pair(s)"] = df.apply(
        lambda row: f'<Patient>: {row["question(s)"]} \n <Therapist>: {row["answer(s)"]}', axis=1
    )
    df = df.apply(lambda col: col.map(lambda x: x.replace("\xa0", " ") if isinstance(x, str) else x))
    df = df.replace(r"\\'", "'", regex=True)
    df.to_csv("../../data/01_preprocessed/pair_data_preproc.csv", index=False)

def preproc_mental_conv():
    df = pd.read_csv("../../data/00_raw/mental_health_conv.csv")
    df = df.rename(
        columns={"Context": "question(s)", "Response": "answer(s)"}
    )
    df["question_answer_pair(s)"] = df.apply(
        lambda row: f'<Patient>: {row["question(s)"]} \n <Therapist>: {row["answer(s)"]}', axis=1
    )
    df = df.apply(lambda col: col.map(lambda x: x.replace("\xa0", " ") if isinstance(x, str) else x))
    df.to_csv("../../data/01_preprocessed/mental_conv_preproc.csv", index=False)

def preproc_synth_convs():
    with open("../../data/00_raw/synthetic_conversations.csv", encoding="utf-8") as input_csv,\
        open("../../data/01_preprocessed/synthetic_conversations_preproc.csv", "w", encoding="utf-8") as output_csv:

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
                            "<Patient>: " + question + "\n" + "<Therapist>: " + answer
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
    preproc_mental_conv()
    preproc_mental_faq()
