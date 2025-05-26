from langchain_community.document_loaders import PyMuPDFLoader



def load_pdf_text(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()
    return " ".join([page.page_content for page in pages])
