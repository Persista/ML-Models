#loading the API key
import getpass
import os
os.environ['HUGGING_FACE_HUB_API_KEY'] = getpass.getpass('Hugging face api key:')
# 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

import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
import langchain
import chromadb

import os
import getpass

from langchain.document_loaders import PyPDFLoader  #document loader: https://python.langchain.com/docs/modules/data_connection/document_loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter  #document transformer: text splitter for chunking
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.vectorstores import Chroma #vector store
from langchain import HuggingFaceHub  #model hub
from langchain.chains import RetrievalQA

from langchain.memory import ConversationBufferMemory

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "4bit/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code = True
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import TextStreamer
streamer = TextStreamer(tokenizer,
                        skip_prompt = True,
                        skip_special_tokens = True,
                        use_multiprocessing = False)

from transformers import GenerationConfig
generation_config = GenerationConfig.from_pretrained(model_name)

from transformers import pipeline
pipe = pipeline("text-generation",
                model=model,
                tokenizer = tokenizer,
                max_length = 2048,
                temperature=0,
                top_p=0.95,
                repetition_penalty = 1.15,
                generation_config = generation_config,
                streamer = streamer,
                batch_size=1,
                do_sample = True)

embeddings = HuggingFaceEmbeddings()

sentiment_pipeline = pipeline(task='sentiment-analysis',model='lxyuan/distilbert-base-multilingual-cased-sentiments-student')

user_ids = set()

## deyummm it worked
from flask import Flask, request, jsonify
from pyngrok import ngrok
from langchain.llms import HuggingFacePipeline
from langchain.docstore.document import Document

# retrieval_chain = None
retrieval_chain_dynamic = None

app = Flask(__name__)
port_no = 5000  # Change this to your desired port number

# Configure the ngrok tunnel directly within the script
ngrok.set_auth_token("2XgO9eCxxYzLz8VVG6unMDJK3Yp_5mpYLwCQNEUwdV8v3fmrJ")

# Define the ngrok tunnel configuration
public_url = ngrok.connect(port_no, proto="http", bind_tls=True)

@app.route('/', methods=['POST'])
def home():
    if request.method == 'POST':

        query = request.form['query']
        context = request.form['context']
        id = request.form['id']
        chat_history = request.form['history']
        instruction = request.form['instruction']
        # chat history must be in the format of {"input": "hi"}, {"output": "whats up"}

        document = Document(page_content=context)  # Use context as the initial document
        pages = [document]

        splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
        docs = splitter.split_documents(pages)
        # embeddings = HuggingFaceEmbeddings()
        doc_search = Chroma.from_documents(docs, embeddings)

        llm_model = HuggingFacePipeline(pipeline=pipe)

        if not instruction:
            instruction = ""
        else:
            instruction = "Follow these instructions "+instruction

        template = f"""
            Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to step by step answer the question of the user.Engage in a conversation and, based on the context, take any necessary actions to address the user's concerns and turn their negative sentiment into a positive one. Begin the conversation and guide it towards a resolution that leaves the user satisfied.
            {instruction}
            ------
            <ctx>
            {{context}}
            </ctx>
            ------
            <hs>
            {{history}}
            </hs>
            ------
            {{question}}
            Answer:
            """

        prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=template,
        )

        if id not in user_ids:
            # add first in the user set
            user_ids.add(id)
            # make new instance of memory
            memory = ConversationBufferMemory(
                memory_key="history",
                input_key="question"
            )

            retrieval_chain = RetrievalQA.from_chain_type(llm_model,
                                                          chain_type='stuff',
                                                          retriever=doc_search.as_retriever(),
                                                          chain_type_kwargs={
                                                              "prompt": prompt,
                                                              "memory": memory
                                                          })
        else:
            # take chat history string directly from the history from the request
            memory = ConversationBufferMemory(
                memory_key = "history",
                input_key = "question"
            )

            memory.save_context(chat_history)

            retrieval_chain = RetrievalQA.from_chain_type(llm_model,
                                                          chain_type='stuff',
                                                          retriever=doc_search.as_retriever(),
                                                          chain_type_kwargs={
                                                              "prompt": prompt,
                                                              "memory": memory
                                                          })

        sentiment_data = sentiment_pipeline(query)
        sentiment_score = sentiment_data[0]['score']
        sentiment_label = sentiment_data[0]['label']

        if sentiment_score > 0.85 and setiment_label=='positive':
            positive_template = f"""
            Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to respond the user with a thanking message for retaining our company benifits.
            ------
            <ctx>
            {{context}}
            </ctx>
            ------
            <hs>
            {{history}}
            </hs>
            ------
            {{question}}
            Answer:
            """

            positive_prompt = PromptTemplate(
                input_variables=["history", "context", "question"],
                template=positive_template,
            )
            positive_retrieval_chain = RetrievalQA.from_chain_type(llm_model,
                                                          chain_type='stuff',
                                                          retriever=doc_search.as_retriever(),
                                                          chain_type_kwargs={
                                                              "prompt": positive_prompt,
                                                              "memory": memory
                                                          })
            positive_response = positive_retrieval_chain.run(query)

            return jsonify({"response":postive_response, "status": 1, "sentiment_score": sentiment_score}) # make it from llm
        elif sentiment_score > 0.85 and sentiment_label=='negative':
            negative_template = f"""
            Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to respond the user with a sorry message that the company could not meet their needs.
            ------
            <ctx>
            {{context}}
            </ctx>
            ------
            <hs>
            {{history}}
            </hs>
            ------
            {{question}}
            Answer:
            """

            negative_prompt = PromptTemplate(
                input_variables=["history", "context", "question"],
                template=negative_template,
            )
            negative_retrieval_chain = RetrievalQA.from_chain_type(llm_model,
                                                          chain_type='stuff',
                                                          retriever=doc_search.as_retriever(),
                                                          chain_type_kwargs={
                                                              "prompt": negative_prompt,
                                                              "memory": memory
                                                          })
            negative_response = negative_retrieval_chain.run(query)
            return jsonify({"response":negative_response, "status": -1, "sentiment_score": sentiment_score})
        else:
            result = retrieval_chain.run(query)  # Call your retrieval_chain.run() function
            return jsonify({"response": result, "status": 0, "sentiment_score": sentiment_score})  # Return a JSON response


if __name__ == "__main__":
    print(f"To access the Global link, please click {public_url.public_url}")
    app.run(port=port_no)
