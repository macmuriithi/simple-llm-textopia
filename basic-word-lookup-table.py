# Welcome to Textopia, where words spark stories!
# This is our first adventure: a simple word prediction model with no learning.
# Our librarian is young and uses a notebook to track which words follow others.
# She predicts the next word by picking the most common one from her notes.

# === Step 1: The Textopia Storybook ===
# The librarian needs a tale to learn from. This is our tiny storybook.
# It’s a short sentence, like a single page in Textopia’s vast library.
text = "once upon a time there was a curious ai who loved to learn"

# === Step 2: Build the Notebook (Word Pair Counts) ===
# To predict words, the librarian counts how often each word follows another.
# This is like writing down pairs (e.g., "once" -> "upon") in her notebook.
# Split the story into words, converting to lowercase for simplicity.
words = text.lower().split()  # E.g., ["once", "upon", "a", ...]
# Create an empty notebook (dictionary) to store word pairs and their counts.
pair_counts = {}
# Loop through the story, pairing each word with the next one.
for i in range(len(words) - 1):  # Stop before the last word to have a next word.
    current_word = words[i]  # E.g., "once".
    next_word = words[i + 1]  # E.g., "upon".
    # If the current word isn’t in the notebook, add it with an empty sub-notebook.
    if current_word not in pair_counts:
        pair_counts[current_word] = {}
    # Increment the count for this word pair (e.g., "once" -> "upon" gets +1).
    pair_counts[current_word][next_word] = pair_counts[current_word].get(next_word, 0) + 1
    # E.g., pair_counts["once"]["upon"] = 1.

# === Step 3: Predict the Next Word ===
# The librarian uses her notebook to predict the next word.
# Given a word, she picks the one that follows it most often.
def predict_next_word(current_word):
    # Check if the word is in the notebook.
    if current_word not in pair_counts:
        return "unknown"  # If not, say "unknown" (like a missing page).
    # Get the sub-notebook of words that follow the current word.
    next_words = pair_counts[current_word]  # E.g., {"upon": 1}.
    # Find the next word with the highest count (most common).
    return max(next_words, key=next_words.get)  # E.g., "upon" if it has the highest count.

# === Step 4: Tell a Short Tale ===
# The librarian starts with a word and predicts the next few words to weave a tale.
# Start with a seed word to begin the story.
start_word = "once"
tale = [start_word]  # The tale starts: ["once"].
# Predict the next 5 words to build the tale.
for _ in range(5):
    # Get the last word in the tale to predict the next one.
    next_word = predict_next_word(tale[-1])  # E.g., predict "upon" after "once".
    # Add the predicted word to the tale.
    tale.append(next_word)  # E.g., tale = ["once", "upon"].
# Join the words into a story and share it.
print("Generated tale:", " ".join(tale))

# Bravo, Textopia explorer! This simple librarian predicts words using her notebook.
# She can’t learn or adapt, but she’s a great start.
# Next, we’ll give her a magical brain to learn patterns dynamically!
