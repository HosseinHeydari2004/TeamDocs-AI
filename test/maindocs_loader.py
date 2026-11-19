from src.rag import LoaderPipeline

loader = LoaderPipeline()
docs = loader.load(
    source=[
        "https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html",
        "vectorstore.py",
        r"D:\AI Source\Projects\list_of_projects\TeamDocs-AI\README.md",
    ]
)
print(len(docs))
