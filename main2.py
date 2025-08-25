import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)


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

	template = """
	Use the following pieces of context to answer the question at the end. \
	If you don't know the answer, just say that you don't know. Don't try to make up an answer. \
	Use maximum 3 sentences and keep the answer as concise as possible. \
	Always say "Thanks for Asking!" at the end of an answer. \
	
	{context}

	Question: {question}

	Helpful Answer:
	"""

	# Prompt created with a custom template
	custom_rag_prompt = PromptTemplate.from_template(template=template)

	# Creating a rag chain with prompt and llm
	rag_chain = (
		{
			"context": vectorstore.as_retriever() | format_docs,
			"question": RunnablePassthrough()
		}
		| custom_rag_prompt
		| llm
	)

	# Invoke the chain
	result = rag_chain.invoke(query)
	print("\n\n ----- Answer")
	print(result.content)