import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Define vocabulary
vocabulary = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '|', ':', '%', ' ', '/', '|:', '|%', '|]', '|:', '|^', '||']
num_symbols = len(vocabulary)

def tokenize_phrase(phrase):
    return [vocabulary.index(symbol) for symbol in phrase if symbol in vocabulary]

def generate_music_from_phrase(phrase, model, sequence_length, num_generated_symbols):
    tokenized_phrase = tokenize_phrase(phrase)
    tokenized_phrase = pad_sequences([tokenized_phrase], maxlen=sequence_length, padding='pre')

    generated_sequence = tokenized_phrase[0].tolist()
    for _ in range(num_generated_symbols):

        seed_sequence_reshaped = np.reshape(generated_sequence, (1, sequence_length, 1))

        predicted_symbol = np.argmax(model.predict(seed_sequence_reshaped), axis=-1)

        generated_sequence.append(predicted_symbol[0])
        generated_sequence = generated_sequence[1:]
    generated_music = ''.join([vocabulary[symbol] for symbol in generated_sequence])
    return generated_music
sample_dataset = """
X: 1
T:Breeze
% Nottingham Music Database
S:Traditional
M:4/4
K:D
D2 FA d2 fd|c2 Ac G2 Bc|d2 f2 e2 c2|B2 d2 A2 F2:|

X: 2
T:Meadow Serenade
% Nottingham Music Database
S:By Mary Smith
M:6/8
K:D
"D"FAA AFA|def "A7"edc|"D"def afa|dfe "A7"edc|
"D"FAA AFA|def "A7"edc|"D"def "A7"a2f|edc "D"d3:|

X: 3
T:Riverside Waltz
% Nottingham Music Database
S:Anonymous
M:3/4
K:G
"G"B2 d2 d2|B2 A2 A2|G2 E2 E2|G2 F2 D2|
"G"B2 d2 d2|B2 A2 A2|G2 E2 E2|D6:|

X: 4
T:Sunset Lullaby
% Nottingham Music Database
S:By John Doe
M:4/4
K:C
"C"CEGE GABc|dcBA G2 z2|CEGE GABc|dcBA G2 z2|

X: 5
T:Moonlight Sonata
% Nottingham Music Database
S:By Ludwig van Beethoven
M:3/4
K:C#
"C#"e3 d c|d3 B c|e3 d c|d3 B c|

X: 6
T:Spring Symphony
% Nottingham Music Database
S:By Jane Doe
M:4/4
K:G
"G"B2 B2 A2 G2|B2 B2 A4|c2 c2 B2 A2|c2 c2 B4|

X: 7
T:Autumn Waltz
% Nottingham Music Database
S:Traditional
M:3/4
K:D
"D"F2 A2 d2|f2 d2 A2|B2 d2 f2|d6|

X: 8
T:Winter Serenade
% Nottingham Music Database
S:By John Smith
M:6/8
K:G
"G"D2 G2 B2|d2 B2 G2|"Em"E2 A2 c2|e2 c2 A2|

X: 9
T:Summer Nocturne
% Nottingham Music Database
S:Anonymous
M:4/4
K:F
"F"c2 c2 c2 B2|A2 A2 A2 G2|F2 F2 F2 E2|D6|


X: 10
T:Forest Song
% Nottingham Music Database
S:By Emily Johnson
M:6/8
K:Am
"Am"A2 A B c|e2 e d c|"G"B2 B d e|"Am"c3 z A B|
"""

tokenized_dataset = [vocabulary.index(symbol) for symbol in sample_dataset if symbol in vocabulary]

sequence_length = 50
sequences = []
labels = []
for i in range(len(tokenized_dataset) - sequence_length):
    sequences.append(tokenized_dataset[i:i+sequence_length])
    labels.append(tokenized_dataset[i+sequence_length])

X = np.array(sequences)
y = np.array(labels)

# One-hot encode labels
y = tf.keras.utils.to_categorical(y, num_classes=num_symbols)

# Define the model architecture with GRU
model = Sequential([
    GRU(units=256, input_shape=(sequence_length, 1)),
    Dense(num_symbols, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, y, batch_size=64, epochs=50)

# Save the trained model
model.save("trained_model_gru.h5")

seed_phrase = input("Enter the Phrase:")
generated_music = generate_music_from_phrase(seed_phrase, model, sequence_length, num_generated_symbols=100)

print("Generated Music:")
print(generated_music)
import music21

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
generated_music_stream = symbolic_to_midi(generated_music)
generated_music_stream.write('midi', fp='generated_music_mee.mid')
