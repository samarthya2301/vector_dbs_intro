import os
from typing import Dict

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == "__main__":

	print("Retrieving ...")

	openai_embeddings = OpenAIEmbeddings()
	llm = ChatOpenAI(model="gpt-4o-mini")
	query = "What is PineCone in Machine Learning? Explain within 50 words."

	# Fetching the custom data from the vector store
	vectorstore = PineconeVectorStore(
		index_name=os.environ['INDEX_NAME'],
		embedding=openai_embeddings
	)

	# Prompt for langchain hub
	retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

	# Chain for passing a list of Documents to the Model (LCEL Runnable)
	combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

	# Pass the vector store and docs chain
	retrival_chain = create_retrieval_chain(
		retriever=vectorstore.as_retriever(),
		combine_docs_chain=combine_docs_chain
	)

	# Invoke the chain with the query
	result: Dict = retrival_chain.invoke(input={"input": query})
	print("\n\n ----- Answer")
	print(result.get("answer"))
	print("\n\n ----- Context Used (from PineCone)")
	print(result.get("context"))