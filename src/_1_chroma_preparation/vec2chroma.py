import chromadb
import csv
import ast
from time import time
from tqdm import tqdm
from src._1_chroma_preparation.chroma_utils import ChromaCollectionManager

def upsert_embeddings(emb_path: str,
                      collection: chromadb.Collection,
                      doc_fieldname: str,
                      embed_fieldname: str):
    
    with open(emb_path, encoding="utf-8") as embedding_csv:
        
        reader = csv.DictReader(embedding_csv)
        
        duplicate_docs = 0
        ids = []
        embeddings = []
        batch = []
        metadatas = []
        batch_size = 500
        inserted_rows = 0
        unique_docs = set()
        
        bar = tqdm()
        start = time()
        for row in reader:

            id = row[doc_fieldname]  # set id as the document itself, not important
            if id in unique_docs:
                duplicate_docs += 1
            else:
                unique_docs.add(id)
                ids.append(id)
                batch.append(row[doc_fieldname])
                embeddings.append(ast.literal_eval(row[embed_fieldname]))

            if len(batch) >= batch_size:
                collection.upsert(
                    ids=ids,
                    documents=batch,
                    embeddings=embeddings,
                )
                inserted_rows += len(batch)
                bar.update(len(batch))
                batch = []
                ids = []
                embeddings = []
                metadatas = []
        
        if batch:
            collection.upsert(
                ids=ids,
                documents=batch,
                embeddings=embeddings,
            )
            inserted_rows += len(batch)
            batch.clear()
            ids.clear()
            embeddings.clear()
            metadatas.clear()
        
        end = time()
        bar.close()
        embedding_csv.close()
        
        print("done!")
        print(f"inserted in total {inserted_rows} docs")
        print(f"insertion duration: {(end - start)/60:.2f}min")

def main():

    pubmed_emb_path = "data/03_embedded/pubmed_abstracts.csv"

    chroma_handler = ChromaCollectionManager(persist_dir="data/chroma")
    pubmed_collection = chroma_handler.create_empty_collection(
        collection_name="pubmed_collection"
    )

    upsert_embeddings(
        emb_path=pubmed_emb_path,
        collection=pubmed_collection,
        doc_fieldname="doc",
        embed_fieldname="embedding"
    )

if __name__ == "__main__":
    main()
