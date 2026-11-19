from rag.loaders import PDFLoader

loader = PDFLoader()
docs = loader.load(
    path=r"C:\Users\hosse\Downloads\NovaTech_Team_Handbook_TEST.pdf"
)
for doc in docs:
    print(doc.page_content)
    print("_" * 80)