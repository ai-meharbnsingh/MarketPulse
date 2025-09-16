# File: pdf_extractor.py

import pdfplumber
import pandas as pd
from pathlib import Path


def extract_pdf_content(pdf_path: str, output_path: str):
    """
    Extracts all text and tables from a PDF document and saves them
    to a structured text file.

    Args:
        pdf_path (str): The full path to the input PDF file.
        output_path (str): The full path to save the extracted text file.
    """
    # Ensure the input file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        print(f"❌ Error: The file '{pdf_path}' was not found.")
        return

    print(f"🚀 Starting extraction from: {pdf_path}")

    # Using 'with' ensures the PDF file is properly closed after processing
    with pdfplumber.open(pdf_file) as pdf:
        # Open the output file in write mode with UTF-8 encoding to handle various characters
        with open(output_path, 'w', encoding='utf-8') as output_file:

            # Write a header to the output file
            output_file.write(f"--- START OF EXTRACTED CONTENT FOR {pdf_file.name} ---\n\n")

            # Loop through each page of the PDF
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                print(f"📄 Processing Page {page_number} of {len(pdf.pages)}...")

                # --- Write Page Header ---
                output_file.write(f"\n\n--- PAGE {page_number} ---\n\n")

                # --- Extract and Write Plain Text ---
                # .extract_text() is a robust method to get the text content of the page
                text = page.extract_text()
                if text:
                    output_file.write("== Text Content ==\n")
                    output_file.write(text)
                    output_file.write("\n")

                # --- Extract and Write Tables ---
                # .extract_tables() finds and extracts all tables on the page
                tables = page.extract_tables()
                if tables:
                    output_file.write("\n== Table Content ==\n")
                    # Loop through each table found on the page
                    for table_num, table_data in enumerate(tables):
                        output_file.write(f"\n--- Table {table_num + 1} on Page {page_number} ---\n")

                        # Use pandas to format the table nicely as a string
                        # This handles missing values (None) gracefully
                        try:
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            # .to_markdown() is a clean, readable format for text files
                            table_md = df.to_markdown(index=False)
                            output_file.write(table_md)
                            output_file.write("\n")
                        except Exception as e:
                            # If formatting fails, write the raw data
                            output_file.write(f"Could not format table with pandas: {e}\n")
                            output_file.write("Raw table data:\n")
                            for row in table_data:
                                # Convert each item in the row to a string to avoid errors
                                # Join with a tab for basic alignment
                                output_file.write('\t'.join(map(str, row)) + '\n')

            # Write a footer to the output file
            output_file.write(f"\n\n--- END OF EXTRACTED CONTENT ---")

    print(f"✅ Success! All content extracted and saved to: {output_path}")


# --- HOW TO USE THIS SCRIPT ---
if __name__ == "__main__":
    # 1. DEFINE the path to the PDF you want to analyze.
    #    - Replace "YOUR_DOCUMENT.pdf" with the actual file name.
    #    - Make sure the PDF file is in the same directory as this script,
    #      or provide the full path to it (e.g., "C:/Users/YourUser/Documents/report.pdf").
    input_pdf_file = "Detailed_Presenttion.pdf"

    # 2. DEFINE the name for the output text file.
    #    - The script will create this file.
    output_text_file = "extracted_content.txt"

    # 3. RUN the script.
    extract_pdf_content(pdf_path=input_pdf_file, output_path=output_text_file)

    # 4. After the script finishes, open "extracted_content.txt",
    #    copy its entire content, and paste it into the chat for analysis.