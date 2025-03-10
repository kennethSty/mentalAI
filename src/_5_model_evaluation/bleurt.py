from src._0_data_preparation.create_conversation_dataloaders import get_conversation_dataloaders
from src._2_chat.chat import init_conversation
from src._5_model_evaluation.evaluation_utils import load_finetuned_model
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.gpu_utils import DeviceManager
from src.config import FINETUNE_EVAL_CONFIG
from pathlib import Path
import logging
from src.utils.paths_utils import set_up_logging
from bleurt import score


def evaluate_model_on_testset(model: nn.Module, test_loader: DataLoader, device: str, model_flag: str):
    set_up_logging(log_file_name=f"{model_flag}_bleurt_eval_log.txt")
    scores = []

    # init bleurt model
    checkpoint = "bleurt/bleurt/BLEURT-20"
    scorer = score.BleurtScorer(checkpoint)

    model.eval()
    for i, (questions, answers) in enumerate(test_loader):
        llm_pipe = init_conversation(device=device)

        gen_answers = []
        for question in questions:
            try:
                answer = llm_pipe.get_answer(question)
            except Exception as e:
                print(e)
                answer = ""

            gen_answers.append(answer)

        batch_scores = scorer.score(references=answers, candidates=gen_answers)

        scores.append(batch_scores)
        log_message = f"\nScores: {batch_scores}"
        logging.info(log_message)
        print(log_message)

    avg_score = np.mean(np.array(scores))

    log_message = f"\nAverage Score: {avg_score}"
    logging.info(log_message)
    print(log_message)



if __name__ == "__main__":
    device = DeviceManager().get_device()
    assert device == "cuda" or device == "mps", "Evaluation has to be done on GPU"

    #Set model flag to choose the right tokenized test datasets
    model_flag = "gpt2"  # gpt2 #emobert, #psychbert
    model, tokenizer = load_finetuned_model(model_flag=model_flag, device=device)

    test_ds_path = Path(f"data/02_train_test_splits/test/counsel_conversations_test.csv")
    _, test_loader, _ = get_conversation_dataloaders(
        batch_size=FINETUNE_EVAL_CONFIG["batch_size"],
        tokenizer=tokenizer,
        test_ds_path=test_ds_path,
    )

    evaluate_model_on_testset(model, test_loader, device, model_flag=model_flag)