from rag.loaders import WebLoader

loader = WebLoader()

docs = loader.load(
    url="https://pytorch.org/get-started/locally/"
)
for doc in docs:
    print(doc.page_content)
    print("//" * 80)
    print(doc.metadata)
    print("_" * 80)
