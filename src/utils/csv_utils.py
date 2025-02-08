import sys
import csv

class CSVUtils:
    @staticmethod
    def increase_csv_maxsize():
        """
        Increases the max csv field size limit to handle large files
        """
        maxInt = sys.maxsize
        while True:
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)
        csv.field_size_limit(maxInt)
        print("Reset csv maxsize to:", maxInt)
