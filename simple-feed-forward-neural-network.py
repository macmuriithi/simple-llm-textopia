import torch  # PyTorch: our spellbook for building the librarian’s magical brain.
import torch.nn as nn  # Neural network tools, like parts of the brain.
import torch.optim as optim  # Optimizers to guide the librarian’s learning.

# Welcome to Textopia, where our librarian is gaining a magical brain!
# This neural network predicts the next word by learning patterns, unlike the static notebook.
# We’re using Python (our scroll) and PyTorch (our spellbook) to make it happen.

# === Step 1: The Textopia Storybook ===
# The librarian needs a tale to learn from. This is our tiny storybook.
# It’s the same short sentence as before, keeping things familiar.
text = "once upon a time there was a curious ai who loved to learn"

# === Step 2: Word Token Spell (Turning Words into Numbers) ===
# Computers don’t understand words, so we cast a spell to turn them into numbers.
# Each unique word gets a code, like a secret key to a treasure chest.
# Split the story into words, all lowercase to keep things simple.
words = text.lower().split()  # E.g., ["once", "upon", "a", ...]
# Find all unique words to create the librarian’s vocabulary.
vocab = sorted(set(words))  # E.g., ["a", "ai", "curious", ...]
# Count how many unique words we have (the size of the vocabulary).
vocab_size = len(vocab)  # E.g., 13 if there are 13 unique words.
# Create a dictionary to map each word to a number (e.g., "once" -> 0).
word_to_idx = {word: i for i, word in enumerate(vocab)}
# Create a reverse dictionary to map numbers back to words (e.g., 0 -> "once").
idx_to_word = {i: word for i, word in enumerate(vocab)}
# Turn the entire story into a list of numbers using the token spell.
data = [word_to_idx[word] for word in words]  # E.g., [0, 1, 2, ...]

# === Step 3: The Librarian’s Brain (Neural Network) ===
# We build a simple neural network, like a magical brain with two parts:
# 1. An embedding layer to turn word numbers into rich portraits (vectors).
# 2. A thinking layer to predict the next word.
class WordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim=10):
        # Initialize the brain, inheriting from PyTorch’s neural network module.
        super(WordPredictor, self).__init__()
        # The embedding layer creates a portrait (vector) for each word.
        # vocab_size: number of unique words; embed_dim: size of each portrait.
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # E.g., 13 words, 10D vectors.
        # The thinking layer (fully connected) predicts the next word.
        # It takes the portrait and outputs scores for each possible word.
        self.fc = nn.Linear(embed_dim, vocab_size)  # E.g., 10D input, 13 outputs.
    
    def forward(self, x):
        # The forward pass: how the brain processes input to make predictions.
        # x is a tensor of word numbers (e.g., [0] for "once").
        x = self.embedding(x)  # Turn numbers into portraits (e.g., [0] -> [0.1, 0.2, ...]).
        x = self.fc(x)  # Predict scores for the next word (e.g., [0.5, -0.2, ...]).
        return x  # Return the prediction scores.

# === Step 4: Prepare Lessons (Training Data) ===
# To teach the librarian, we create pairs of words: current word and next word.
# These are like flashcards for the librarian to study.
# Inputs: all words except the last one (e.g., "once", "upon", ...).
inputs = torch.tensor([data[i] for i in range(len(data) - 1)], dtype=torch.long)
# Targets: the words that follow (e.g., "upon", "a", ...).
targets = torch.tensor([data[i + 1] for i in range(len(data) - 1)], dtype=torch.long)

# === Step 5: Train the Brain (Learning Loop) ===
# The librarian studies the flashcards to improve her predictions.
# Create the brain with our WordPredictor class.
model = WordPredictor(vocab_size, embed_dim=10)  # Small brain: 10D embeddings.
# Use cross-entropy loss to measure how wrong the predictions are.
# It’s like grading the librarian’s guesses (lower score = better).
criterion = nn.CrossEntropyLoss()
# Use Adam optimizer to guide the librarian’s learning, like a wise teacher.
# It adjusts the brain’s connections (weights) to reduce errors.
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Learning rate: how fast to learn.
# Study for 100 rounds (epochs), like practice sessions.
for epoch in range(100):
    # Clear old notes to start fresh each round.
    optimizer.zero_grad()  # Reset gradients (like erasing a chalkboard).
    # Feed the inputs (word numbers) to the brain and get predictions.
    outputs = model(inputs.unsqueeze(0)).squeeze(0)  # unsqueeze: add a batch dimension.
    # Compare predictions to targets and calculate the error (loss).
    loss = criterion(outputs, targets)  # Loss: how far off the guesses were.
    # Tell the brain where it went wrong (backpropagation).
    loss.backward()  # Compute gradients for weights.
    # Update the brain’s connections to improve next time.
    optimizer.step()  # Adjust weights based on gradients.
    # Every 20 rounds, share progress to see how the librarian’s doing.
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")  # Loss should decrease.

# === Step 6: Tell a Tale (Generate Text) ===
# Now the librarian can predict words to weave a short tale.
# Set the brain to storytelling mode (no learning).
model.eval()
# Start with a seed word to begin the tale.
start_word = "once"
tale = [start_word]  # The tale starts: ["once"]
# Get the number code for the starting word.
current_idx = word_to_idx[start_word]  # E.g., "once" -> 0.
# Predict the next 5 words to build the tale.
with torch.no_grad():  # No learning, just predicting.
    for _ in range(5):
        # Turn the current word’s number into a tensor (PyTorch’s magical array).
        input_tensor = torch.tensor([current_idx], dtype=torch.long)
        # Feed it to the brain to get prediction scores.
        output = model(input_tensor)
        # Pick the word with the highest score (most likely next word).
        next_idx = torch.argmax(output, dim=-1).item()  # E.g., 1 for "upon".
        # Turn the number back into a word.
        next_word = idx_to_word[next_idx]  # E.g., 1 -> "upon".
        # Add the word to the tale.
        tale.append(next_word)  # E.g., tale = ["once", "upon"].
        # Update the current word for the next prediction.
        current_idx = next_idx  # E.g., now use "upon" (1).
# Join the words into a story and share it.
print("Generated tale:", " ".join(tale))

# Bravo, Textopia explorer! The librarian’s brain is learning to predict words.
# She only sees one word at a time, but it’s a big step from the notebook.
# Next, we’ll add magical glasses (attention) to see whole sequences!
