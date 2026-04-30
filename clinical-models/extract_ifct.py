"""
Extract IFCT 2017 nutritional data from PDF - Focus on data tables
"""
import pdfplumber
import pandas as pd
import re
from pathlib import Path

def extract_ifct_tables(pdf_path: str, output_file: str):
    """Extract nutritional data tables from IFCT PDF"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Opening PDF: {pdf_path}\n")
        
        with pdfplumber.open(pdf_path) as pdf:
            f.write(f"Total pages: {len(pdf.pages)}\n\n")
            
            # Table 1: Proximate Principles starts around page 31 (PDF offset)
            # Let's check pages 35-60 for actual data tables
            for i in range(34, 80):  # Pages 35-80 (0-indexed)
                if i >= len(pdf.pages):
                    break
                    
                page = pdf.pages[i]
                
                f.write(f"\n{'='*80}\n")
                f.write(f"PAGE {i+1}\n")
                f.write(f"{'='*80}\n")
                
                # Extract text
                text = page.extract_text()
                if text:
                    f.write("TEXT:\n")
                    f.write(text[:3000] if len(text) > 3000 else text)
                    f.write("\n\n")
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    f.write(f"TABLES: {len(tables)} table(s) found\n")
                    for j, table in enumerate(tables):
                        f.write(f"\n--- Table {j+1}: {len(table)} rows, {len(table[0]) if table else 0} cols ---\n")
                        if table and len(table) > 0:
                            # Write first 10 rows
                            for row_idx, row in enumerate(table[:10]):
                                f.write(f"  Row {row_idx}: {row}\n")
                            if len(table) > 10:
                                f.write(f"  ... {len(table) - 10} more rows\n")
        
        f.write("\nExtraction complete.\n")
    
    print(f"Analysis saved to: {output_file}")

if __name__ == "__main__":
    pdf_path = Path(__file__).parent.parent / "IFCT2017.pdf"
    output_file = Path(__file__).parent / "ifct_tables_analysis.txt"
    
    print(f"PDF Path: {pdf_path}")
    print(f"Output: {output_file}")
    
    extract_ifct_tables(str(pdf_path), str(output_file))
