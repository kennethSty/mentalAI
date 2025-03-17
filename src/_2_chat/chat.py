import asyncio

from src._2_chat.ChatPipeline import ChatPipeline
from src.utils.gpu_utils import DeviceManager
from src._1_chroma_preparation.embed_utils import EmbeddingFunction, MiniLML6, PubMedBert
from src._2_chat.prompts import get_prompt
from src.utils.UI import print_greeting, print_farewell, print_separator, print_newline_separator

from langchain.prompts import PromptTemplate

def chat():
    device_handler = DeviceManager()
    device = device_handler.get_device()

    top_k = 2
    collection_embed_dict = init_collections()
    prompt = PromptTemplate(
        template=get_prompt(),
        input_variables=["top_k_abstracts", "top_k_conversations", "suicide_risk", "user_query"]
    )
    llm_pipe = ChatPipeline(
        top_k=top_k,
        model_path="../../models/pretrained/llama-2-7b-chat.Q5_K_M.gguf",
        chroma_path="../../data/chroma",
        collection_embed_dict = collection_embed_dict,
        prompt = prompt,
    )

    print_greeting()
    question = input("Your question: ")
    while (question.strip().lower() != "/exit"):
        print_separator()
        llm_pipe.get_answer(question)
        print_newline_separator()
        question = input("Your question: ")
    print_farewell()


def init_collections():
    device = DeviceManager().get_device()

    pubmed_embedder = PubMedBert(device=device)
    conversation_embedder = MiniLML6(device=device)

    pubmed_embed_fn = EmbeddingFunction(pubmed_embedder)
    conv_embed_fn = EmbeddingFunction(conversation_embedder)

    collection_embed_dict = {
        "pubmed_collection": pubmed_embed_fn,
        "conv_collection": conv_embed_fn
    }

    return collection_embed_dict


if __name__ == "__main__":
    chat()
