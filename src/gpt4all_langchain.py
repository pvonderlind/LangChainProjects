from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import os
from src import config

callbacks = [StreamingStdOutCallbackHandler()]

llm = GPT4All(model=config.gpt_4all_groovy, backend='gptj', callbacks=callbacks, verbose=False)
embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: "
)

if not os.path.exists('phb_faiss_index'):
    print('Re-Indexing Document')
    loader = PyPDFLoader(
        "C:\\Users\\phili\\Documents\\GitHub\\University\\Master\\Books\\Machine Learning\\RLbook2020.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=32)
    pages = text_splitter.split_documents(documents)

    faiss_index = FAISS.from_documents(pages, embeddings)
    faiss_index.save_local('phb_faiss_index')
else:
    faiss_index = FAISS.load_local('phb_faiss_index', embeddings)

llm_qa_chain = load_qa_chain(llm=llm, chain_type='stuff')


def get_answer_for_question(question: str):
    similar_documents = faiss_index.similarity_search(question)
    return llm_qa_chain.run({'input_documents': similar_documents, 'question': question})


while True:
    try:
        print(get_answer_for_question(input("Enter your question:")))
    except KeyboardInterrupt:
        break
