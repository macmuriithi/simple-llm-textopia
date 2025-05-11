import torch
import torch.nn as nn
import torch.optim as optim
import math

# Welcome to Textopia, a magical land where words flow like rivers!
# You're an explorer building a wise librarian (our LLM) who can understand and create language.
# This code is your spellbook to bring the librarian to life.

# === Step 1: The Textopia Library (Our Training Data) ===
# The librarian needs books to learn from. This is our tiny library—a short story snippet.
# In Textopia, this text is the river of words the librarian will study to learn patterns.
text = """Once upon a time, there was a curious AI who loved to learn and explore the world of code and language."""

# === Step 2: The Token Spell (Turning Words into Numbers) ===
# Computers don’t understand words, so we cast a spell to turn characters into numbers.
# Each unique character (like 'a', 'b', or ' ') gets a secret code, like a magical handshake.
# This is tokenization, where we create a dictionary to map characters to numbers and back.
chars = sorted(list(set(text)))  # Find all unique characters in our library
vocab_size = len(chars)  # How many unique characters we have
char_to_idx = {ch: i for i, ch in enumerate(chars)}  # Map character to number (e.g., 'a' -> 0)
idx_to_char = {i: ch for i, ch in enumerate(chars)}  # Map number back to character (e.g., 0 -> 'a')

# Turn the entire text into a list of numbers using our token spell.
# This is what the librarian will work with instead of raw text.
data = [char_to_idx[ch] for ch in text]

# === Step 3: The Librarian’s Brain (The Transformer Model) ===
# Here, we build the librarian’s magical brain—a transformer model.
# It has special glasses (attention) to see how characters relate and a memory to predict the next one.
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, dim_feedforward=128, max_seq_length=50):
        super(SimpleTransformer, self).__init__()
        # The embedding layer turns number tokens into rich descriptions (vectors).
        # Think of it as giving each character a personality profile.
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding adds a sense of order, like page numbers in a book.
        # It helps the librarian know which character comes first, second, etc.
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        # The transformer encoder is the librarian’s core magic: attention and thinking layers.
        # Attention lets it focus on related characters, like noticing 'c' and 'a' form 'cat'.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # The final layer turns the librarian’s thoughts back into character predictions.
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x):
        # Turn tokens into vectors and boost them for clarity (like sharpening the glasses).
        x = self.embedding(x) * math.sqrt(self.d_model)
        # Add positional information so the librarian knows the sequence order.
        x = self.pos_encoder(x)
        # Use attention to connect related characters and think deeply about them.
        x = self.transformer_encoder(x)
        # Predict the next character by mapping back to our vocabulary.
        x = self.fc(x)
        return x

# === Step 4: The Position Spell (Keeping Track of Order) ===
# Since the librarian reads sequences, it needs to know which character comes where.
# This positional encoding is like a magical ruler that marks each character’s place.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a map of positions using math magic (sines and cosines).
        # This map helps the librarian remember the order of characters.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the position map to the character vectors, like stamping page numbers.
        x = x + self.pe[:, :x.size(1), :]
        return x

# === Step 5: Preparing for School (Training Data Batches) ===
# To train the librarian, we give it small chunks of text to study, like homework assignments.
# Each chunk is a sequence of characters, and the task is to guess the next character.
def get_batch(data, seq_length=20, batch_size=32):
    # Pick random starting points in our text to create mini-lessons.
    idx = torch.randint(0, len(data) - seq_length, (batch_size,))
    # Create input sequences (what the librarian reads).
    x = torch.stack([torch.tensor(data[i:i+seq_length]) for i in idx])
    # Create target sequences (what the librarian should predict next).
    y = torch.stack([torch.tensor(data[i+1:i+seq_length+1]) for i in idx])
    return x, y

# === Step 6: The Librarian’s School (Training the Model) ===
# Now, the librarian goes to school to learn patterns by guessing the next character.
# It gets feedback (loss) and adjusts its brain (weights) to improve.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer(vocab_size=vocab_size).to(device)
criterion = nn.CrossEntropyLoss()  # Measures how wrong the guesses are.
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Helps the librarian learn smarter.

# Training loop: the librarian’s study sessions.
num_epochs = 100  # Number of study rounds.
seq_length = 20  # Length of each text chunk.
batch_size = 32  # Number of chunks per study session.

for epoch in range(num_epochs):
    model.train()  # Put the librarian in learning mode.
    x, y = get_batch(data, seq_length, batch_size)  # Get a new homework batch.
    x, y = x.to(device), y.to(device)  # Send it to the magical study room (GPU or CPU).
    
    optimizer.zero_grad()  # Clear old notes before a new lesson.
    output = model(x)  # The librarian makes predictions.
    loss = criterion(output.view(-1, vocab_size), y.view(-1))  # Check how wrong the predictions were.
    loss.backward()  # Tell the librarian where it went wrong.
    optimizer.step()  # Help the librarian improve its guesses.
    
    if (epoch + 1) % 10 == 0:  # Every 10 rounds, check progress.
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# === Step 7: Storytelling Time (Generating Text) ===
# Now that the librarian is trained, it can tell stories by predicting the next character.
# You give it a starting sentence, and it weaves a tale, like improv in Textopia.
def generate_text(model, start_text, max_length=50):
    model.eval()  # Put the librarian in storytelling mode.
    # Cast the token spell to turn the starting text into numbers.
    current_seq = [char_to_idx[ch] for ch in start_text]
    generated = current_seq.copy()  # Keep a copy for the final story.
    
    with torch.no_grad():  # No learning, just storytelling.
        for _ in range(max_length - len(start_text)):
            # Use the last few characters as input (like a memory limit).
            input_seq = torch.tensor([current_seq[-seq_length:]], dtype=torch.long).to(device)
            output = model(input_seq)  # The librarian predicts the next character.
            next_token_logits = output[0, -1, :]  # Get the prediction scores.
            next_token = torch.argmax(next_token_logits, dim=-1).item()  # Pick the most likely character.
            generated.append(next_token)  # Add it to the story.
            current_seq.append(next_token)  # Update the memory.
    
    # Turn the number sequence back into text using the reverse token spell.
    return ''.join([idx_to_char[idx] for idx in generated])

# === Step 8: Unleash the Librarian’s Tale ===
# Let’s start with a seed phrase and see what story the librarian tells!
start_text = "Once upon"
generated_text = generate_text(model, start_text)
print(f"Generated text: {generated_text}")

# Congratulations, explorer! You’ve built a piece of Textopia’s magic.
# The librarian is small, but it’s learning. To make it wiser, give it more books (data),
# more study time (epochs), or stronger glasses (bigger model). Keep exploring!
