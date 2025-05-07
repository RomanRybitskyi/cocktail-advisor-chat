from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import pandas as pd

class CocktailVectorStore:
    def __init__(self, path='data/final_cocktails.csv'):
        self.df = pd.read_csv(path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vectorstore = None

    def prepare_documents(self):
        docs = []
        for _, row in self.df.iterrows():
            content = f"{row['name']} - {'Alcoholic' if row['alcoholic'] else 'Non-Alcoholic'}\nIngredients: {row['ingredients']}\nInstructions: {row['instructions']}"
            docs.append(Document(page_content=content, metadata={"name": row["name"]}))
        return docs

    def build_vectorstore(self):
        documents = self.prepare_documents()
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts)
        self.vectorstore = FAISS.from_texts(texts, embedding=self.model.encode)
        return self.vectorstore

    def search(self, query, k=5):
        return self.vectorstore.similarity_search(query, k=k)
