# Simple-LLM-Textopia

Welcome to **Simple-LLM-Textopia**, a beginner-friendly journey into the magical land of language models! In Textopia, you’ll build a tiny librarian—a character-level transformer built with PyTorch—that learns to predict and weave tales from text. With story-driven comments, this project demystifies tokenization, attention, and training, making it perfect for explorers new to LLMs. Step into Textopia and bring a piece of its word-woven magic to life!

## About the Project

In Textopia, words flow like rivers, and our librarian (the LLM) learns their patterns to tell stories. This project implements a simple transformer model that:
- Turns text into numbers (tokenization).
- Uses attention to connect characters, like magical glasses spotting patterns.
- Trains to predict the next character in a sequence.
- Generates new text, crafting tales from a starting prompt.

Built for beginners, the code is annotated with comments that guide you through each step as if you’re adventuring in Textopia. It’s a small but enchanting step into the world of natural language processing!

## Prerequisites

To explore Textopia, you’ll need:
- **Python 3.x** (3.7 or higher recommended).
- **PyTorch**: Install via `pip install torch` (works on CPU or GPU).
- A text editor or IDE (e.g., VS Code, Jupyter Notebook).
- Basic familiarity with Python (don’t worry, the comments will guide you!).

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Simple-LLM-Textopia.git
   cd Simple-LLM-Textopia
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch
   ```

3. **Verify Setup**:
   Ensure `character-level-transformer.py` is in the repository. This is the spellbook containing the librarian’s code!

## Usage

Run the code to train the librarian and hear its tale:
```bash
python character-level-transformer.py
```

### What Happens?
- The librarian reads a tiny story (the training data).
- It trains for 100 study sessions (epochs), learning to predict the next character.
- It generates text starting with “Once upon,” weaving a short tale.
- Progress is printed every 10 epochs, showing the librarian’s learning (loss).

### Example Output
Due to the small dataset, the output might be simple or repetitive, like:
```
Generated text: Once upon a timmmmeeee, therrrrrr...
```
With more data or training, the librarian’s stories will grow wiser!

## Adventures
- 1st Adventure `basic-word-lookup-table.py` :  A simple model that predicts the next word based on word pair frequencies, like a dictionary of “what follows what.” It uses no machine learning, just counting how often one word follows another. It’s a static approach, like a lookup table, with no learning or adaptation, making it an intuitive first step to understand word prediction in LLMs.
- 2nd Adventure `simple-feed-foward-neural-network.py` : A simple feed-forward neural network that learns to predict the next word using embeddings and a single layer, introducing machine learning and neural networks.
- `character-level-transformer.py`: The script implemented with a character level transfomer and Textopia-themed comments.
- `word-level-transformer.py`: The  script implemented with a word level transformer and Textopia-themed comments 
- `README.md`: This file, your guide to Textopia.

## Future Enhancements

Textopia is just beginning! Potential adventures include:
- Adding a larger library (dataset) for richer stories.
- Upgrading to word or subword tokenization for smarter predictions.
- Strengthening the librarian’s brain with more transformer layers.
- Creating a demo notebook for interactive exploration.

## Contributing

Fellow explorers are welcome to join Textopia! Feel free to:
- Submit issues for bugs or ideas.
- Fork the repo and add your own Textopia twists.
- Share your generated tales or improvements!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Inspired by the magic of language models and the joy of learning through storytelling. Built with ❤️ for beginners stepping into the world of AI.

---

**Ready to explore Textopia?** Run the code, follow the librarian’s journey, and let the words flow!
