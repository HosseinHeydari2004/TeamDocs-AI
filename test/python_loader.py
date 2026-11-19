from rag.loaders import PythonCodeLoader

loader = PythonCodeLoader()

docs = loader.load(
    path=r"/rag/loaders\base.py"
)
print(docs[0].page_content)