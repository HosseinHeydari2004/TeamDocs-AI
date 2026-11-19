from src.rag import PreprocessingPipeline

preprocessor = PreprocessingPipeline()
docs = preprocessor.run(
    source=[
        "vectorstore.py",
    ]
)
for doc in docs:
    print(doc)
    print("_" * 80)
