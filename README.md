# MentalAI (Work in progress)

MentalAI is a retrieval-augmented system for mental health support with fine-tuned suicide prediction. It leverages **llama.cpp** and **LangChain** to provide AI-assisted conversations based on retrieved contextual information.

## Folder Structure

The repository is structured as follows:

```
mentalAI/
│── src/
│   ├── _0_data_preparation/       # Data processing pipeline
│   ├── _1_chroma_preparation/     # Setting up the vector store using ChromaDB
│   ├── _2_chat/                   # Chat system logic
│   ├── utils/                     # Utility functions
│   ├── __init__.py
```

## Setup & Usage (More details follow soon)

Since the dataset for this project is private, users need to manually set up all required data resources. After preparing the data:

1. **Set up the Chroma vector store** following the scripts in `src/_1_chroma_preparation/`.
2. **Run the chat system** by executing:
   ```bash
   python src/_2_chat/chat.py
   ```

## Technologies Used

- **llama.cpp** - Running LLaMA models efficiently on CPUs.
- **LangChain** - Managing AI-driven conversations and retrieval augmentation.
- **ChromaDB** - Vector store for efficient embedding retrieval.

## Disclaimer

This project is intended for research and educational purposes. It is not a replacement for professional mental health services.

