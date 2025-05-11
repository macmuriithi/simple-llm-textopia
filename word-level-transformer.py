import torch
import torch.nn as nn
import torch.optim as optim
import math

# Welcome to Textopia, where words weave stories like threads in a magical tapestry!
# You're crafting a wise librarian (our LLM) who now thinks in whole words, not just characters.
# This code is your enchanted scroll, bringing a word-level language model to life.

# === Step 1: The Textopia Library (Our Storybook) ===
# The librarian needs a tale to learn from. This is our small storybook—a snippet of text.
# In Textopia, these words are the sparkling gems the librarian will study to uncover patterns.
text = """Once upon a time, there was a curious AI who loved to learn and explore the world of code and language."""

# === Step 2: The Word Token Spell (Turning Words into Magic Codes) ===
# To work with words, we cast a powerful spell to turn each unique word into a number.
# This is word-level tokenization, where every word gets a secret code, like a key to a treasure chest.
words = text.lower().split()  # Break the story into words, all lowercase for consistency.
vocab = sorted(set(words))  # Find all unique words in our tale.
vocab_size = len(vocab)  # Count the unique words (our vocabulary size).
word_to_idx = {word: i for i, word in enumerate(vocab)}  # Map each word to a number (e.g., 'once' -> 0).
idx_to_word = {i: word for i, word in enumerate(vocab)}  # Map numbers back to words (e.g., 0 -> 'once').

# Turn the story into a sequence of numbers using our word token spell.
# This is what the librarian will read and learn from.
data = [word_to_idx[word] for word in words]

# === Step 3: The Librarian’s Mind (The Transformer Model) ===
# Here, we forge the librarian’s enchanted mind—a transformer that understands word connections.
# It has magical glasses (attention) to see how words relate and a memory to predict the next word.
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, dim_feedforward=128, max_seq_length=50):
        super(SimpleTransformer, self).__init__()
        # The embedding layer turns word codes into rich portraits (vectors), giving each word a personality.
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding adds a sense of order, like bookmarks showing where each word sits in the story.
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        # The transformer encoder is the librarian’s core magic: attention to connect words and deep thinking to understand them.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # The final layer turns the librarian’s thoughts into predictions for the next word.
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x):
        # Turn word codes into vectors and amplify them for clarity (like polishing the magical glasses).
        x = self.embedding(x) * math.sqrt(self.d_model)
        # Add positional bookmarks so the librarian knows the word order.
        x = self.pos_encoder(x)
        # Use attention to link related words (e.g., 'once' and 'upon') and think deeply about the sequence.
        x = self.transformer_encoder(x)
        # Predict the next word by mapping back to our vocabulary.
        x = self.fc(x)
        return x

# === Step 4: The Bookmark Spell (Tracking Word Order) ===
# Words need to be in the right order, like pages in a book. This spell adds position markers.
# Positional encoding helps the librarian remember which word comes first, second, and so on.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a magical map of positions using waves of math (sines and cosines).
        # This map marks each word’s place in the sequence.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Stamp the position map onto the word vectors, like adding page numbers to a story.
        x = x + self.pe[:, :x.size(1), :]
        return x

# === Step 5: Preparing for Story School (Training Data Batches) ===
# To teach the librarian, we give it short word sequences, like chapters from our storybook.
# Each chapter is a lesson to predict the next word in the tale.
def get_batch(data, seq_length=5, batch_size=4):
    # Pick random starting points in the story to create mini-chapters.
    idx = torch.randint(0, len(data) - seq_length, (batch_size,))
    # Create input sequences (the words the librarian reads).
    x = torch.stack([torch.tensor(data[i:i+seq_length]) for i in idx])
    # Create target sequences (the next words the librarian must predict).
    y = torch.stack([torch.tensor(data[i+1:i+seq_length+1]) for i in idx])
    return x, y

# === Step 6: The Librarian’s Story School (Training the Model) ===
# Now, the librarian attends school to learn the art of storytelling by guessing the next word.
# Feedback (loss) helps it refine its mind (weights) to weave better tales.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer(vocab_size=vocab_size).to(device)
criterion = nn.CrossEntropyLoss()  # Measures how wrong the librarian’s guesses are.
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Guides the librarian to learn smarter.

# Training loop: the librarian’s study sessions.
num_epochs = 200  # Number of lessons to master storytelling.
seq_length = 5  # Length of each word sequence (a short chapter).
batch_size = 4  # Number of chapters per lesson.

for epoch in range(num_epochs):
    model.train()  # Set the librarian to learning mode.
    x, y = get_batch(data, seq_length, batch_size)  # Fetch a new batch of story chapters.
    x, y = x.to(device), y.to(device)  # Send them to the enchanted study hall (GPU or CPU).
    
    optimizer.zero_grad()  # Clear old notes before a new lesson.
    output = model(x)  # The librarian predicts the next words.
    loss = criterion(output.view(-1, vocab_size), y.view(-1))  # Check how wrong the predictions were.
    loss.backward()  # Show the librarian its mistakes.
    optimizer.step()  # Help the librarian improve its storytelling.
    
    if (epoch + 1) % 20 == 0:  # Every 20 lessons, share progress.
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# === Step 7: Weaving New Tales (Generating Text) ===
# With training complete, the librarian can now tell stories by predicting the next word.
# Give it a starting phrase, and it crafts a tale, like a bard in Textopia’s great hall.
def generate_text(model, start_text, max_length=20):
    model.eval()  # Set the librarian to storytelling mode.
    # Cast the word token spell to turn the starting phrase into numbers.
    words = start_text.lower().split()
    current_seq = [word_to_idx.get(word, 0) for word in words]  # Use 0 for unknown words.
    generated = current_seq.copy()  # Keep a copy for the final tale.
    
    with torch.no_grad():  # No learning, just tale-weaving.
        for _ in range(max_length - len(current_seq)):
            # Use the last few words as input, like a memory of the story so far.
            input_seq = torch.tensor([current_seq[-seq_length:]], dtype=torch.long).to(device)
            output = model(input_seq)  # The librarian predicts the next word.
            next_token_logits = output[0, -1, :]  # Get the prediction scores.
            next_token = torch.argmax(next_token_logits, dim=-1).item()  # Choose the most likely word.
            generated.append(next_token)  # Add it to the tale.
            current_seq.append(next_token)  # Update the story memory.
    
    # Turn the number sequence back into words, like lifting the token spell.
    return ' '.join([idx_to_word[idx] for idx in generated])

# === Step 8: Hear the Librarian’s Tale ===
# Let’s give the librarian a starting phrase and listen to its story!
start_text = "once upon a time"
print("Generated text:", generate_text(model, start_text))

# Bravo, Textopia explorer! You’ve conjured a word-wise librarian.
# This model is small, but its tales will grow with more stories (data),
# longer lessons (epochs), or a mightier mind (bigger model). Keep weaving magic!
