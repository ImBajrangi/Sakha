# Sakha AI: Vrindopnishad Personal Assistant

Sakha is a specialized AI assistant designed for **Vrindopnishad**, featuring a hybrid Deep Learning architecture for semantic retrieval and text generation.

## 🏗 Project Structure

- `assistant.py`: Main entry point (interactive CLI).
- `core/`: Core logic modules.
    - `chatbot.py`: Central orchestrator and matching logic.
    - `searcher.py`: Web search integration via DuckDuckGo.
    - `synthesizer.py`: Local LLM (GPT-2) response generation.
- `data/`: Knowledge base storage.
    - `dataset.json`: High-quality expert knowledge and user feedback.
- `scripts/`: Training and synchronization.
    - `local_tune.py`: Fine-tunes the local model on Mac (MPS support).
    - `colab_train.py`: Standalone script for GPU training in Google Colab.
- `tests/`: Verification scripts for various features.
- `config.py`: Centralized configuration settings.

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install torch transformers sentence-transformers duckduckgo-search
   ```

2. **Run the Assistant**:
   ```bash
   python assistant.py
   ```

## 🧠 Self-Learning (RLHF System)

Sakha learns from your interactions in real-time:
- **Teach**: Use `/correct [correct info]` to fix an answer.
- **Reinforce**: Use `/good` to save a successful web answer as native knowledge.
- **Synchronize**: Use `/train` to bake your feedback directly into the model's neural weights.

## 🛠 Advanced Architectures
- **Semantic Retrieval**: Uses `Sentence-BERT` (Sentence-Transformers) for high-precision intent matching.
- **Local Generation**: Uses a fine-tuned `GPT-2` model for context-aware responses, optimized for Mac Silicon (MPS).
- **Domain-Specific Search**: Intelligent fallback with region-locked (India) search and automatic domain expansion.

Radhe Radhe! 🙏
