from rag.loaders import DocsLoader

loader = DocsLoader()

docs = loader.load(
    path=r"C:\Users\hosse\Downloads\TeamDocs AI – Test Documentation.docx"
)

for doc in docs:
    print(doc.page_content)
    print("_" * 80)