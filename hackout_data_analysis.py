from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

from flask import Flask, request, jsonify
from pyngrok import ngrok
from sklearn.metrics.pairwise import cosine_similarity

def get_embeddings(embedding_model,words):
    embeddings = embedding_model.encode(words)
    df = pd.DataFrame(embeddings,columns = [f"feat_{i}" for i in range(384)])
    return df

threshold = 0.7  # Adjust as needed

def combine_rows_based_on_cosine_similarity(df, threshold):
    num_rows = df.shape[0]
    combined = set()

    for i in range(num_rows):
        if i not in combined:
            for j in range(i + 1, num_rows):
                if j not in combined:
                    i_vector = df.iloc[i, :].values.reshape(1, -1)
                    j_vector = df.iloc[j, :].values.reshape(1, -1)
                    sim = cosine_similarity(i_vector, j_vector)[0][0]
                    if sim > threshold:
                        combined.add(j)

    return df.drop(index=list(combined))

def combine_similar_words_to_unique(result_df,embedding_df,words):
    unique_words = set()
    for sentence, embedding in zip(words, embedding_df.values):
        exists = (result_df==embedding).all(axis=1).any()

        if exists:
            unique_words.add(sentence)

    unique_words = list(unique_words)
    return unique_words

from transformers import pipeline
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np
import json

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)

def measure_user_intent(query,extractor):
    # Load pipeline

    keyphrases = extractor(f"""{query}""".replace("\n", " "))
    return keyphrases

app = Flask(__name__)
port_no = 3000  # Change this to your desired port number

# Configure the ngrok tunnel directly within the script
ngrok.set_auth_token("2Xi0Us7DYzDIYk1I5wjI2Ky9aKd_7bXnTTPPy2cT5znmK9zY5")

# Define the ngrok tunnel configuration
public_url = ngrok.connect(port_no, proto="http", bind_tls=True)

@app.route('/', methods=['POST'])
def home():

    if request.method == 'POST':
        query = request.form['query']
        preexisting_words = request.form['words']

        prex_words = preexisting_words.split(";")

        words = measure_user_intent(query,extractor).tolist()
        embedings_df = get_embeddings(embedding_model,words)
        prex_embeddings_df = get_embeddings(embedding_model,prex_words)
        total_df = pd.concat([prex_embeddings_df,embeddings_df])
        result_df = combine_rows_based_on_cosine_similarity(total_df, threshold)
        unique_words = combine_similar_words_to_unique(result_df,embeddings_df,words)

        return jsonify({"result":unique_words})

if __name__ == "__main__":
    print(f"To access the Global link, please click {public_url.public_url}")
    app.run(port=port_no)













