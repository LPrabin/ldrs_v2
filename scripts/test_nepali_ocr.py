import logging
import os
import sys

# Add the project root to sys.path
sys.path.append(os.getcwd())

from ldrs.pdf_extractor import PdfExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    pdf_path = "tests/docs/Annual-Report-2079-80-Nepali.pdf"
    output_dir = "tests/results_test"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    extractor = PdfExtractor(pdf_path=pdf_path, output_dir=output_dir)
    print("Starting extraction...")
    md_path = extractor.extract()
    print(f"Extraction complete. Markdown saved to: {md_path}")
    
    # Check if tables file was created
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    table_path = os.path.join(output_dir, f"{stem}_tables.md")
    if os.path.exists(table_path):
        print(f"Tables saved to: {table_path}")
    else:
        print("No tables file created.")

if __name__ == "__main__":
    main()
