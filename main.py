from flask import Flask, render_template, request, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import music21

app = Flask(__name__)

# Define vocabulary
vocabulary = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '|', ':', '%', ' ', '/', '|:', '|%', '|]', '|:', '|^', '||']
num_symbols = len(vocabulary)

def tokenize_phrase(phrase):
    return [vocabulary.index(symbol) for symbol in phrase if symbol in vocabulary]

def symbolic_to_midi(symbolic_music):
    stream = music21.stream.Stream()

    for symbol in symbolic_music:

        if symbol not in ('A', 'B', 'C', 'D', 'E', 'F', 'G', '|'):
            continue


        if symbol == '|':
            note_or_rest = music21.note.Rest()
        else:
            note_or_rest = music21.note.Note(symbol)

        stream.append(note_or_rest)

    return stream

# Function to generate music based on a given phrase
def generate_music_from_phrase(phrase, model, sequence_length, num_generated_symbols):
    # Tokenize the input phrase
    tokenized_phrase = tokenize_phrase(phrase)

    tokenized_phrase = pad_sequences([tokenized_phrase], maxlen=sequence_length, padding='pre')

    # Generate music based on the input phrase
    generated_sequence = tokenized_phrase[0].tolist()
    for _ in range(num_generated_symbols):
        seed_sequence_reshaped = np.reshape(generated_sequence, (1, sequence_length, 1))

        predicted_symbol = np.argmax(model.predict(seed_sequence_reshaped), axis=-1)
        generated_sequence.append(predicted_symbol[0])
        generated_sequence = generated_sequence[1:]

    generated_music = ''.join([vocabulary[symbol] for symbol in generated_sequence])
    return generated_music

# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\ss\\OneDrive\\Desktop\\meenalmusic\\trained_model_gru.h5")

sequence_length = 50

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        seed_phrase = request.form['seed_phrase']
        generated_music = generate_music_from_phrase(seed_phrase, model, sequence_length, num_generated_symbols=100)

        # Convert generated music to MIDI format
        generated_music_stream = symbolic_to_midi(generated_music)
        generated_music_stream.write('midi', fp='static/generated_music_mee.mid')

        return render_template('index.html', generated_music=generated_music)
    return render_template('index.html', generated_music=None)

@app.route('/download')
def download_file():
    return send_file('static/generated_music_mee.mid', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
