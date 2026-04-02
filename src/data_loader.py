from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
import os


def pdf_loader(pdfs_path="data"):
    all_docs=[]
    pdf_dir=Path(pdfs_path)

    pdf_files=list(pdf_dir.glob('**/*.pdf'))

    print(f"Found {len(pdf_files)} pdf files for processing ")

    for files in pdf_files:
        print(f"\n Processing {files.name}")

        try:
            loader=PyMuPDFLoader(str(files))
            documents=loader.load()

            for document in documents:
                document.metadata['source_file']=files.name
                document.metadata['type']="pdf"

            all_docs.extend(documents)

            print(f"Total of {len(documents)} pages loaded")

        except Exception as e:
            print(f"An error occured while data parsing: {e}")

    
    print(f"\n Total documents loaded = {len(all_docs)}")
    return all_docs
    


