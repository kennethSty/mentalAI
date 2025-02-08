from tqdm import tqdm
from typing import Dict, List
from Bio import Entrez
from datetime import datetime, timedelta
import io
import csv
import random
import logging

from src.utils.csv_utils import CSVUtils

class DataExtractor:
    def __init__(self, query_term = "mental health",
                window_duration_days = 7,
                chunk_size = 100,
                start_date = '2019/01/01',
                end_date = '2025/01/01',
                save_path = "../../data/pubmed_abstracts.csv",
                log_path = "../../logs/pubmed_extraction_log.txt"):
        
        """
        Class encapsulating the logic of retrieving data from PubMed

        :param query_term: The search term for which pubmed articles are retrieved
        :param window_duration_days: The size of a time window for which articles are retrieved
        :param chunk_size: The size of chunks in which articles are retrieved
        :param start_date: The date from which onwards query matches are retrieved
        :param end_date: The date until which query matches are retrieved
        """
        
        self.query_term = query_term
        self.window_duration_days = window_duration_days
        self.chunk_size = chunk_size
        self.start_date = start_date
        self.end_date = end_date
        self.save_path = save_path
        self.log_path = log_path

    def extract(self):
        """
        Function that calls the PubMed API via the Entrez package to extract data matching
        the extract parameters specified in self.extract_params.

        :return:
            pandas dataframe containing the crawled details (abstracts, keywords, etc.) of all
            articles matching the query.
        """

        # Set up logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Output to console
                logging.FileHandler(self.log_path, mode='a', encoding="utf-8")  # Output to file
            ]
        )

        logging.info("------- Beginning to extract PubMed abstracts --------")
        logging.info(f"Start date: {self.start_date}")
        logging.info(f"End date: {self.end_date}")
        logging.info(f"Query term: {self.query_term}")

        total_docs_processed = 0
        abstracts_missing = 0

        # Increase capacity to handle large files
        CSVUtils.increase_csv_maxsize()

        with open(self.save_path, "w", encoding="utf-8") as output_csv:
            writer = csv.DictWriter(output_csv, fieldnames=["title", "abstract", "author", "year"])
            writer.writeheader()

            studiesIdList = self.__get_article_IDs()  # Fetch article IDs
            logging.info(f"Found {len(studiesIdList)} matches.")

            for chunk_i in tqdm(range(0, len(studiesIdList), self.chunk_size)):  # Process in chunks
                chunk = studiesIdList[chunk_i:chunk_i + self.chunk_size]
                papers = self.fetch_details(chunk, chunk_i)
                rows_to_write = []

                for i, paper in enumerate(papers['PubmedArticle']):
                    try:
                        title = paper['MedlineCitation']['Article']['ArticleTitle']
                    except:
                        title = 'NA'
                    try:
                        abstract = paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                    except:
                        abstract = 'NA'
                        abstracts_missing += 1
                    try:
                        author = [", ".join([author.get('LastName'), author.get('ForeName')]) for author in
                                  paper['MedlineCitation']['Article']['AuthorList']]
                    except:
                        author = 'NA'
                    try:
                        year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year']
                    except:
                        year = 'NA'

                    row = {"title": title, "abstract": abstract, "author": author, "year": year}
                    rows_to_write.append(row)

                writer.writerows(rows_to_write)
                total_docs_processed += len(chunk)

                # Log every 5th chunk
                if chunk_i % 10 == 0:
                    logging.info(f"** Year of last extraction: {year}")


        logging.info("--------")
        logging.info("Data extraction finished.")
        logging.info(f"Total abstracts retrieved: {total_docs_processed}")
        logging.info(f"Number of missing abstracts: {abstracts_missing}")

        return

    def fetch_details(self, id_list, chunk_i) -> Dict:
        """
        Function to fetch detailed data for a list of ID's of articles in Pubmed

        :param
            id_list: list of IDs from esearch
        :return:
            nested Dictionary containing the detailed data (in XML)
        """
        ids = ','.join(id_list)
        email = 'emails@examples3.com'
        if chunk_i % 100000:
            email = f'emails@examples{chunk_i}.com'
        Entrez.email = email
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=ids)
        results = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                xml_data = handle.read()
                results = Entrez.read(io.BytesIO(xml_data), validate=False)
                break  # Break out of the loop if successful
            except Exception as e:
                 logging.info(f"--- In fetch details --- Error on attempt {attempt + 1}: {e}")
        else:
            logging.info("--- In fetch details --- Failed after maximum retries.")
        handle.close()
        return results

    def __get_article_IDs(self) -> List:
        """
        Function that extracts article IDs for later fetching in a batch-wise manner as specified by a window.
        :return
            list of IDs
        """
        result_dicts = {}
        start_date = datetime.strptime(self.start_date, '%Y/%m/%d')
        end_date = datetime.strptime(self.end_date, '%Y/%m/%d')
        window_duration = timedelta(days=self.window_duration_days)  # timedelta(days=30)
        current_date = start_date
        window_end = start_date + window_duration

        # Loop over time windows 
        last_iteration = False
        while window_end <= end_date:
            returned_dicts = self.__search(
                'Disease', 
                current_date.strftime('%Y/%m/%d'),
                window_end.strftime('%Y/%m/%d')
            )

            # accumulate dictionary values
            for key, value in returned_dicts.items():
                if key in result_dicts:
                    if isinstance(value, list):
                        if isinstance(result_dicts[key], list):
                            # If both are lists, extend the existing list with the new list
                            result_dicts[key].extend(value)
                        else:
                            # If the existing value is not a list, create a new list with both values
                            result_dicts[key] = [result_dicts[key]] + value
                    else:
                        if isinstance(result_dicts[key], list):
                            # If the existing value is a list, append the new value to it
                            result_dicts[key].append(value)
                        else:
                            # If neither is a list, create a list with both values
                            result_dicts[key] = [result_dicts[key], value]
                else:
                    # Add the key-value pair to result_dicts
                    result_dicts[key] = value
            current_date = window_end
            window_end = current_date + window_duration
            if last_iteration:
                break
            elif window_end > end_date:
                window_end = end_date
                last_iteration = True

        return result_dicts['IdList']
    
    def __search(self, query, mindate, maxdate) -> Dict:
        """
        Function to access IDs to queried data using Entrez
        :param
            query: term for which to search
            mindate: end of time period to be extracted
            maxdate: start of time period to be extracteed
        :return:
            Dictionary with the following keys: 'Count', 'RetMax', 'RetStart', 'IdList', 'TranslationSet', 'QueryTranslation'
        """
        # docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch
        nums = []
        for _ in range(15):
            nums.append(str(random.randint(0, 9)))
        num = "".join(nums)
        Entrez.email = f'emails@examples{num}.com'
        handle = Entrez.esearch(db='pubmed',
                                sort='relevance',
                                retmax='10000',
                                retmode='xml',
                                term=query,
                                mindate=mindate,
                                maxdate=maxdate)
        results = Entrez.read(handle)
        return results

def main():
    extractor = DataExtractor()
    studies = extractor.extract()
    studies.to_csv("data/studies.csv", encoding="utf-8", index=False)
    
if __name__ == "__main__":
    main()
    