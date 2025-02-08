import ast
import numpy as np
from typing import Dict, List
from collections.abc import Mapping

class Document(Mapping):
    """ Represents a single document with various metadata fields"""
    def __init__(self, row: Dict):
        self.document_dict = self.preprocess_row(row)

    def __getitem__(self, key):
        return self.document_dict[key]

    def __iter__(self):
        return iter(self.document_dict)

    def __len__(self):
        return len(self.document_dict)

    def preprocess_row(self, row: Dict):
        """Preprocesses a row from csv dictionary reader and cleans missing values"""
        for key in row.keys():
            if type(row[key]) != str:
                if np.isnan(row[key]):
                    row[key] = "NA"
                else:
                    row[key] = str(row[key])

        row["author"] = self.format_authors(authors_str= row["author"])
        return row

    def format_authors(self, authors_str: str) -> List[str]:
        """ Converts a string into a list of formatted author names """
        if authors_str != "NA":
            authors_str = ast.literal_eval(authors_str)
            author_list = []
            for author in authors_str:
                if "," in author:
                    lastname, firstname = author.split(", ")
                    fullname = firstname + " " + lastname
                else:
                    fullname = author
                author_list.append(fullname)
            authors_str = author_list
        else:
            authors_str = []
        return authors_str

    def get_combined_doc(self) -> str:
        """ Returns a formatted document string. """
        combined_doc = []
        for key, value in self.document_dict.items():
            if not value:
                value = "NA"
            elif type(value) == list:
                value = ", ".join(value)
            combined_doc.append(key + ": " + value)
        return "\n".join(combined_doc)

    def get_doc_info(self) -> str:
        """ Returns document details excluding the abstract. """
        doc_info = ""
        for key, value in self.row.items():
            if key != "abstract":
                if not value:
                    value = "NA"
                elif type(value) == list:
                    value = ", ".join(value)
                doc_info += key + ": " + value + "\n"
        return doc_info

