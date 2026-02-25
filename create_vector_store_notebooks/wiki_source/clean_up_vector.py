from vector_store_module import Retriever

retriever = Retriever()
retriever.clean_empty_documents()
retriever.save()
