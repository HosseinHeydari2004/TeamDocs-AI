from src.rag.rag_pipeline import RagPipeline
from src.rag.rag_agent.rag_agent_workflow import print_workflow
# source = [
#     r"D:\AI Source\Projects\list_of_projects\TeamDocs-AI\src\rag\preprossing\preprossor.py",
#     r"D:\جزوه و اسلاید های و کتاب ها C++\[FVCP9504] C_Plus_Plus_Programming [Files_And_Codes]\S09\10.cpp",
#     r"C:\Users\hosse\Downloads\TeamDocs AI – Test Documentation.docx"
# ]
pipeline = RagPipeline()

# pipeline.add_documents(source=source)

response = pipeline.run_pipeline(
    query="In 10_1.cpp, what would tellg() print if the array a had 10 elements?"
)

print(response["messages"][-1].content)

print("_" * 80)

print_workflow(response=response)


