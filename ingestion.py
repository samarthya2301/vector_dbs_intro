import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":

	# Load the text document into memory
	#  .load() will return a langchain document list
	print("Ingesting ...")
	loader = TextLoader("./mediumblog1.txt")
	document = loader.load()

	# Splitting the text
	#  chunk_size is the size in characters of a chunk
	#  chunk_overlap is the common part between 2 chunks to keep continuity
	print("Splitting ...")
	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
	texts = text_splitter.split_documents(document)
	print(f"Chunks Created: {len(texts)}")

	print("Embedding ...")
	openai_embeddings = OpenAIEmbeddings()
	# openai_embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
	PineconeVectorStore.from_documents(texts, openai_embeddings, index_name=os.environ['INDEX_NAME'])

	print("Finished")