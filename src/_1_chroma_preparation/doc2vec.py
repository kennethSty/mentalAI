import csv
from time import time
from typing import List
from abc import ABC, abstractmethod

from src.utils.csv_utils import CSVUtils
from src._1_chroma_preparation.embed_utils import EmbedModelWrapper, PubMedBert, MiniLML6
from src._1_chroma_preparation.row_read_utils import Document
from src.utils.gpu_utils import DeviceManager

class BaseEmbedder(ABC):

    BATCH_SIZE = 2  # 128 + 64 + 32 (Precomputed for optimal CUDA utilization)

    def __init__(self, model:EmbedModelWrapper, input_doc_path: str, output_emb_path:str):
        self.model = model
        self.input_doc_path = input_doc_path
        self.output_emb_path = output_emb_path
        self.abstract_lookup = set()
        self.total_docs_processed, self.duplicates, self.abstracts_missing, self.invalid_years = 0, 0, 0, 0

    @abstractmethod
    def create_embeddings(self):
        pass

    @abstractmethod
    def _embed_and_write_batch(self, writer, year_batch: List[str], doc_batch: List[str]):
        """Method to be overridden for embedding and writing the batch."""
        pass

    def _print_finish(self, elapsed_time):
        print(f"done!\nembedded in total {self.total_docs_processed} docs\ntotal time: {elapsed_time:.2f}min")
        print(f"found {self.duplicates} duplicate docs")
        print(f"found {self.abstracts_missing} missing abstracts")
        print(f"found {self.invalid_years} years not possible to cast into int")


class PubMedEmbedder(BaseEmbedder):

    def __init__(self, model: EmbedModelWrapper, input_doc_path: str, output_emb_path: str):
        super().__init__(model, input_doc_path, output_emb_path)

    def create_embeddings(self):
        """Processes the CSV and creates embeddings."""
        CSVUtils.increase_csv_maxsize()
        start = time()

        with open(self.input_doc_path, encoding="utf-8") as input_csv, \
             open(self.output_emb_path, "w", encoding="utf-8") as output_csv:

            reader = csv.DictReader(input_csv)
            writer = csv.DictWriter(output_csv, fieldnames=["year", "doc", "embedding"])
            writer.writeheader()

            doc_batch, year_batch = [], []

            for row in reader:
                doc = Document(row)
                scanned_doc = self.__scan_doc(doc)
                if scanned_doc:
                    doc_string, year = scanned_doc
                    doc_batch.append(doc_string)
                    year_batch.append(year)

                if len(doc_batch) >= self.BATCH_SIZE:
                    self.__embed_and_write_batch(
                        writer = writer,
                        year_batch = year_batch,
                        doc_batch = doc_batch
                    )
                    self.total_docs_processed += self.BATCH_SIZE
                    print(f"Embedded {self.total_docs_processed} docs")
                    year_batch.clear()
                    doc_batch.clear()

            if doc_batch:
                self.__embed_and_write_batch(writer, year_batch=year_batch, doc_batch=doc_batch)
                self.total_docs_processed += len(doc_batch)
                year_batch.clear()
                doc_batch.clear()

        end = time()
        elapsed_time = (end - start) / 60
        self._print_finish(elapsed_time)

    def _scan_doc(self, doc: Document):
        """
        Scans a Document object for missing information and tracks it
        :param doc: the input document
        :return: a docstring and year tuple if the Document contains the expected data. None else
        """
        try:
            year = int(doc["year"])
        except ValueError:
            self.invalid_years += 1
            return None

        if doc["abstract"] == "NA":
            self.abstracts_missing += 1
            return None

        if doc["abstract"] in self.abstract_lookup:
            self.duplicates += 1
            return None

        self.abstract_lookup.add(doc["abstract"])
        doc_string = doc.get_combined_doc()
        return doc_string, year

    def _embed_and_write_batch(self, writer: csv.DictWriter, doc_batch: List[str], year_batch: List[str]):
        """Embeds a batch of documents based on the docstring"""
        embeddings = self.model.encode(doc_batch)
        rows_to_write = [
            {"year": year, "doc": doc, "embedding": embedding}
            for year, doc, embedding in zip(year_batch, doc_batch, embeddings)
        ]
        writer.writerows(rows_to_write)


class ConversationEmbedder(BaseEmbedder):
    def __init__(self, model: EmbedModelWrapper, input_doc_path: str, output_emb_path: str):
        super().__init__(model, input_doc_path, output_emb_path)

    def create_embeddings(self):
        """Processes the CSV and creates embeddings."""
        CSVUtils.increase_csv_maxsize()
        start = time()

        with open(self.input_doc_path, encoding="utf-8") as input_csv, \
             open(self.output_emb_path, "w", encoding="utf-8") as output_csv:

            reader = csv.DictReader(input_csv)
            writer = csv.DictWriter(output_csv, fieldnames=["question(s)", "question_answer_pair(s)", "embedding"])
            writer.writeheader()

            qa_pair_batch = []
            question_batch = []

            for row in reader:
                qa_pair_batch.append(row["question_answer_pair(s)"])
                question_batch.append(row["question(s)"])

                if len(qa_pair_batch) > self.BATCH_SIZE:
                    self._embed_and_write_batch(
                        writer=writer,
                        qa_pair_batch=qa_pair_batch,
                        question_batch=question_batch
                    )
                    qa_pair_batch.clear()
                    question_batch.clear()

            if qa_pair_batch:
                self._embed_and_write_batch(
                    writer=writer,
                    qa_pair_batch=qa_pair_batch,
                    question_batch=question_batch
                )
                qa_pair_batch.clear()
                question_batch.clear()

            end = time()
            elapsed_time = (end - start) / 60
            self._print_finish(elapsed_time)


    def _embed_and_write_batch(self, writer: csv.DictWriter, qa_pair_batch: List[str], question_batch: List[str]):
        """Embeds a batch of documents based on the docstring"""
        embeddings = self.model.encode(question_batch)
        rows_to_write = [
            {"question(s)": questions, "question_answer_pair(s)": qa_pairs, "embedding": embedding}
            for questions, qa_pairs, embedding in zip(question_batch, qa_pair_batch, embeddings)
        ]
        writer.writerows(rows_to_write)


def main():
    device = DeviceManager().get_device()
    #pubmed_emb_model = PubMedBert(device=device)
    counsel_emb_model = MiniLML6(device=device)
    #pubmed_embedder = PubMedEmbedder(
        #model = pubmed_emb_model,
        #input_doc_path ="../../data/02_merged/train/pubmed_abstracts.csv",
        #output_emb_path ="../../data/03_embedded/embedded_abstracts.csv"
    #)
    counsel_embedder = ConversationEmbedder(
        model = counsel_emb_model,
        input_doc_path ="../../data/02_merged/train/counsel_conversations_train.csv",
        output_emb_path ="../../data/03_embedded/embedded_conversations.csv"
    )
    #pubmed_embedder.create_embeddings()
    counsel_embedder.create_embeddings()


if __name__ == "__main__":
    main()