import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import logging

class TextSimilarityAnalyzer:
    def __init__(self):
        """
        Initialize the TextSimilarityAnalyzer.

        The Excel file path is retrieved from the 'EXCEL_FILE_PATH' environment variable.
        """
        # Validate and store the absolute path to the Excel file
        self.excel_file_path = self.validate_excel_path(os.getenv("EXCEL_FILE_PATH"))
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        
        # Initialize logger for structured logging
        self.logger = self.setup_logger()

    def setup_logger(self):
        """
        Set up a logger for structured logging.

        Returns:
            logging.Logger: Configured logger.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create a file handler and set the formatter
        file_handler = logging.FileHandler('text_similarity.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        return logger

    def validate_excel_path(self, path):
        """
        Validate and return the absolute path to the Excel file.

        Args:
            path (str): Path to the Excel file.

        Returns:
            str: Absolute path to the Excel file.

        Raises:
            FileNotFoundError: If the specified Excel file is not found.
            ValueError: If the path is not an Excel file.
        """
        try:
            # Ensure the path exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            # Ensure the path points to a file
            if not os.path.isfile(path):
                raise ValueError(f"Not a file: {path}")

            # Ensure the file has a valid Excel extension
            _, extension = os.path.splitext(path)
            if extension.lower() not in ['.xlsx', '.xls']:
                raise ValueError(f"Not a valid Excel file: {path}")

            # Return the absolute path
            return os.path.abspath(path)

        except Exception as e:
            # Log and re-raise the exception
            self.logger.error(f"Error validating Excel path: {str(e)}")
            raise

    def calculate_similarity(self, answer1, answer2):
        """
        Calculate cosine similarity between two text answers.

        Args:
            answer1 (str): First text answer.
            answer2 (str): Second text answer.

        Returns:
            float: Cosine similarity between the two text answers.
        """
        try:
            answer1 = str(answer1).lower() if not pd.isnull(answer1) else ""
            answer2 = str(answer2).lower() if not pd.isnull(answer2) else ""

            if answer1 and answer2:
                tfidf_matrix = self.vectorizer.fit_transform([answer1, answer2])
                cosine_sim = linear_kernel(tfidf_matrix[0], tfidf_matrix[1])[0, 0]
                return cosine_sim
            else:
                return 0.0

        except Exception as e:
            # Log and re-raise the exception
            self.logger.error(f"Error calculating similarity: {str(e)}")
            raise

    def process_sheet(self, sheet_name, dataframe):
        """
        Process a single sheet in the Excel file and calculate similarities.

        Args:
            sheet_name (str): Name of the sheet.
            dataframe (pd.DataFrame): DataFrame containing the sheet data.

        Returns:
            dict: Dictionary containing the calculated results for the sheet.
        """
        try:
            results = {
                "Sheet Name": [],
                "Row": [],
                "Cosine Similarity": [],
                "Result": []
            }

            column1 = "Active Voice"  # Replace with the actual column name
            column2 = "Passive Voice"  # Replace with the actual column name

            for index, row in dataframe.iterrows():
                cosine_sim = self.calculate_similarity(row[column1], row[column2])

                if cosine_sim > 0.5:
                    result = "Passed"
                else:
                    result = "Failed"

                results["Sheet Name"].append(sheet_name)
                results["Row"].append(index + 1)
                results["Cosine Similarity"].append(cosine_sim)
                results["Result"].append(result)

            return results

        except Exception as e:
            # Log and re-raise the exception
            self.logger.error(f"Error processing sheet '{sheet_name}': {str(e)}")
            raise

    def analyze_excel_file(self):
        try:
            xls = pd.ExcelFile(self.excel_file_path)
            all_results = []

            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                sheet_results = self.process_sheet(sheet_name, df)
                all_results.append(sheet_results)

            results_df = pd.DataFrame({
                key: [item for sublist in [result[key] for result in all_results] for item in sublist]
                for key in all_results[0]
            })

            output_excel_file = self.excel_file_path.replace(".xlsx", "-results.xlsx")

            with pd.ExcelWriter(output_excel_file, engine='xlsxwriter') as writer:
                results_df.to_excel(writer, sheet_name="Results", index=False)

            # Log the successful completion of the analysis
            self.logger.info(f"Results saved to: {output_excel_file}")

        except Exception as e:
            # Log and re-raise the exception
            self.logger.error(f"An error occurred during analysis: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Retrieve the Excel file path from the 'EXCEL_FILE_PATH' environment variable
        input_excel_file = os.getenv("EXCEL_FILE_PATH")
        if input_excel_file is None:
            input_excel_file = input("Enter the path to the Excel file: ")

        analyzer = TextSimilarityAnalyzer()
        analyzer.analyze_excel_file()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
