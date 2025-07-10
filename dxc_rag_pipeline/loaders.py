import pytesseract
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    CSVLoader
)

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def get_document_loader(file_path: str):
    
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == "csv":
        return CSVLoader(file_path=file_path, source_column="listing_id")
    
    elif file_extension in ["pdf", "docx", "doc", "png", "jpg", "jpeg"]:
        return UnstructuredFileLoader(file_path)
        
    elif file_extension in ["xlsx", "xls"]:
        from langchain_community.document_loaders import UnstructuredExcelLoader
        return UnstructuredExcelLoader(file_path, mode="elements")
        
    else:
        raise ValueError(f"Unsupported file type: .{file_extension}")
