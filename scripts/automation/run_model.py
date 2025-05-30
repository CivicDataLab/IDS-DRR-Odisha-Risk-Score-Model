import subprocess
import pandas as pd
import unittest
import logging
import sys
from io import StringIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to run a script and capture the output
def run_script(script_path):
    try:
        logging.info(f"Running script: {script_path}")
        result = subprocess.run(["python", script_path], capture_output=True, text=True, check=True)
        logging.info(f"Output of {script_path}: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_path}: {e.stderr}")
        sys.exit(1)

# Test class for data validation
class DataPipelineTests(unittest.TestCase):

    def test_data_shape(self, df, expected_shape):
        """Test if the DataFrame has the expected shape."""
        self.assertEqual(df.shape, expected_shape, f"Expected shape {expected_shape}, but got {df.shape}")

    def test_no_missing_values(self, df):
        """Test if the DataFrame contains missing values."""
        missing_values = df.isnull().sum().sum()
        self.assertEqual(missing_values, 0, f"Data contains {missing_values} missing values")

    def test_column_names(self, df, expected_columns):
        """Test if the DataFrame has expected column names."""
        self.assertListEqual(list(df.columns), expected_columns, "Column names do not match the expected values")

# Function to validate the output data at each stage
def validate_output(data_path, expected_shape, expected_columns):
    try:
        # Read the output data
        df = pd.read_csv(data_path)
        # Capture the test output
        test_output = StringIO()
        test_runner = unittest.TextTestRunner(stream=test_output)
        test_suite = unittest.TestSuite()

        # Adding tests
        test_suite.addTest(DataPipelineTests('test_data_shape', df, expected_shape))
        test_suite.addTest(DataPipelineTests('test_no_missing_values', df))
        test_suite.addTest(DataPipelineTests('test_column_names', df, expected_columns))

        # Run tests
        test_runner.run(test_suite)
        # Display the results
        logging.info(test_output.getvalue())

    except Exception as e:
        logging.error(f"Validation failed: {e}")

# Define the expected shapes and columns at each step
expected_shapes_columns = [
    {"path": "output_step1.csv", "shape": (100, 5), "columns": ["col1", "col2", "col3", "col4", "col5"]},
    {"path": "output_step2.csv", "shape": (200, 4), "columns": ["col1", "col2", "col3", "col4"]},
    # Add more expected shapes and columns as needed
]

# List of scripts to run
scripts = [
    "script1.py",
    "script2.py",
    # Add more scripts as required
]

def main():
    # Run each script and validate the output
    for i, script in enumerate(scripts):
        run_script(script)

        # Validate output after each step
        validate_output(
            data_path=expected_shapes_columns[i]["path"],
            expected_shape=expected_shapes_columns[i]["shape"],
            expected_columns=expected_shapes_columns[i]["columns"]
        )

if __name__ == "__main__":
    main()
