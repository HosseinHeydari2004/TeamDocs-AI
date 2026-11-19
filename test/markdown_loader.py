from rag.loaders import MarkdownLoader

loader = MarkdownLoader()
docs = loader.load(
    path=r"D:\AI Source\Projects\list_of_projects\medvision\README.md"
)
for i, doc in enumerate(docs):
    print(f"doc: {i}")
    print(doc.page_content, end="\n")
    print("_" * 80)