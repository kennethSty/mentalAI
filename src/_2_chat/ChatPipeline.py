from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict

from src._1_chroma_preparation.embed_utils import EmbeddingFunction
from src._1_chroma_preparation.chroma_utils import ChromaCollectionManager

class ChatPipeline:
    def __init__(
            self,
            model_path: str,
            chroma_path: str,
            collection_embed_dict: Dict[str, EmbeddingFunction],
            prompt_template: PromptTemplate
    ):
        self.model = self.__init_llm(model_path=model_path)
        self.chain =  LLMChain(prompt=prompt_template, llm=self.model)
        self.collection_dict = self.__init_collections(
            chroma_path=chroma_path,
            collection_embed_dict=collection_embed_dict
        )


    def __init_collections(self, chroma_path: str, collection_embed_dict):
        collection_manager = ChromaCollectionManager(chroma_path)
        collection_dict = {
            collection_name:
            collection_manager.get_chroma_collection(
                collection_name=collection_name, embed_fn=embed_fn
            ) for collection_name, embed_fn in collection_embed_dict.items()
        }

        return collection_dict

    def __init_llm(
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