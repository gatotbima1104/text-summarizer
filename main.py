from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from heapq import nlargest

app = Flask(__name__)
CORS(app)

def nltk_extractive_summarization(text, num_sentences):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))

    # Calculate word frequency
    word_freq = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word not in stop_words:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

    # Calculate sentence scores based on word frequency
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_freq:
                if i not in sentence_scores:
                    sentence_scores[i] = word_freq[word]
                else:
                    sentence_scores[i] += word_freq[word]

    # Get the top N sentences with highest scores
    top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Sort the top sentences in their original order
    summary = ' '.join([sentences[j] for j in sorted(top_sentences)])

    return summary

@app.route('/', methods=['GET'])
def hello():
    return 'Hello World!'

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    num_sentences = data.get('num_sentences')
    
    summary = nltk_extractive_summarization(text, num_sentences)
    
    response = {
        'summary': summary,
        'sentences': num_sentences
    }
    
    return jsonify(response)

if __name__ == '__main__':
    nltk.download('stopwords')
    app.run()
