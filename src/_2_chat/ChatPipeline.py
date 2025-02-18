import torch
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from typing import Dict

from src._1_chroma_preparation.embed_utils import EmbeddingFunction
from src._1_chroma_preparation.chroma_utils import ChromaCollectionManager
from src._3_model_preparation.emobert_architecture.EmoBertClassifier import EmoBertClassifier
from src._3_model_preparation.gpt_architecture.GPTClassifier import GPTClassifier
from src._3_model_preparation.psychbert_architecture.PsychBertClassifier import PsychBertClassifier
from src.utils.gpu_utils import DeviceManager


class ChatPipeline:
    def __init__(
            self,
            top_k: int,
            model_path: str,
            chroma_path: str,
            collection_embed_dict: Dict[str, EmbeddingFunction],
            prompt: PromptTemplate,
    ):
        self.top_k = top_k
        self.model = self.__init_llm(model_path=model_path)
        self.chain =  prompt | self.model
        self.collection_dict = self.__init_collections(
            chroma_path=chroma_path,
            collection_embed_dict=collection_embed_dict
        )
        self.suicide_classifier = self.__init_suicide_classifier(model_name = "gpt2")

    def get_answer(self, question: str):
        suicide_risk = self.suicide_classifier.classify(question)
        top_pubmed_docs = self.collection_dict["pubmed_collection"]\
            .max_marginal_relevance_search(question, k=self.top_k)
        top_conv_docs = self.collection_dict["conv_collection"]\
            .max_marginal_relevance_search(question, k=self.top_k)

        top_k_abstracts = [doc.page_content for doc in top_pubmed_docs]
        top_k_conversations = [doc.page_content for doc in top_conv_docs]

        top_k_abstracts = "\n\n".join(top_k_abstracts)
        top_k_conversations = "\n\n".join(top_k_conversations)

        answer = self.chain.invoke({
                "top_k_abstracts": top_k_abstracts,
                "top_k_conversations": top_k_conversations,
                "suicide_risk": suicide_risk,
                "user_query": question
        })
        return answer

    def __init_suicide_classifier(self, model_name: str):
        device = DeviceManager().get_device()
        if model_name == "gpt2":
            model = GPTClassifier().to(device)
            checkpoint = torch.load("../../models/finetuned/gpt2_checkpoints/checkpoint_step_8000.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
        elif model_name == "psychbert":
            model = PsychBertClassifier().to(device)
            #TODO: Load from checkpoint
        else:
            assert model_name == "emobert", "model_flag should be emobert"
            model = EmoBertClassifier().to(device)
            #TODO: Load form checkpoint

        return model


    def __init_collections(
            self,
            chroma_path: str,
            collection_embed_dict):
        collection_manager = ChromaCollectionManager(chroma_path)
        collection_dict = {
            collection_name:
            collection_manager.get_chroma_collection(
                collection_name=collection_name, embed_fn=embed_fn
            ) for collection_name, embed_fn in collection_embed_dict.items()
        }

        return collection_dict

    def __init_llm(
            self,
            model_path: str,
            temperature=0.3,
            max_tokens=1000,
            n_ctx=2048,
            top_p=1,
            n_gpu_layers=-1,
            n_batch=512,
            verbose=True,
    ):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            top_p=top_p,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            callback_manager=callback_manager,
            verbose=verbose,  # Verbose is required to pass to the callback manager
        )
        llm.client.verbose = False
        print("\n\n\n")
        return llm