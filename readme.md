# RAG-based Conversational Bot for Comparisons

## Goal
The goal of this project is to create a Retrieval-Augmented Generation (RAG) based conversational bot that:
1. Accepts the names of items to be compared from the user.
2. Crawls the internet to gather related articles and text data.
3. Adds the gathered data to a vector store.
4. Generates a comparison table based on the data.
5. Answers user queries related to the comparison after displaying the table.

As a first step, the current system vectorizes data from a given text document, answers questions based on vector searches, and compares the vectors of user queries with stored file data in a vector store.

## Current Status
- The system can:
  - Vectorize text data from a file.
  - Store the vectorized data in a Qdrant vector store.
  - Use vector search to retrieve context-relevant data.
  - Answer user queries based on vector similarity.
- Technologies used:
  - **Qdrant**: Vector database for storing and retrieving embeddings.
  - **Ollama (Llama 3.1)**: Large Language Model for text generation.
  - **LangChain**: Framework for chaining prompts and outputs.
  - **FastEmbed**: Embedding generator for vectorizing text.

## Setup and Run Instructions

### Prerequisites
1. **Install Dependencies**:
   - Set up a Python virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
     ```
   - Install required Python libraries:
     ```bash
     pip install -r requirements.txt
     ```

2. **Install and Run Qdrant**:
   - Pull the image:
     ```bash
     docker pull qdrant/qdrant
     ```
   - Start the container:
     ```bash
     docker run -p 6333:6333 -p 6334:6334 \
     -v $(pwd)/qdrant_storage:/qdrant/storage:z \
     qdrant/qdrant
     ```
   - For detailed instructions, visit the [Qdrant Quickstart Guide](https://qdrant.tech/documentation/quickstart/).


3. **Install and Run Ollama**:
   - Download and install Ollama from [ollama.com](https://ollama.com/).
   - Run the Llama 3.1 model:
     ```bash
     ollama run llama3.1
     ```
   - On the first run, this will download the model and start the instance.

### Configuration
- Set environment variables and constants in `config.py`.

### Setup the Qdrant Collection
- Run the following script to create a new Qdrant collection (only required during the initial setup):
  ```bash
  python setup_qdrant-collection.py
  ```

### Running the Program
- Once the Qdrant collection is created, run the main program:
  ```bash
  python main.py
  ```

## Project Structure
- `config.py`: Holds environment variables and configuration constants.
- `main.py`: Entry point for the program.
- `setup_qdrant-collection.py`: Script to create a Qdrant collection (run only once).

## Future Steps
1. Integrate web crawling to collect comparison data for user-specified items.
2. Add functionality to generate and display a comparison table.
3. Expand the bot's capabilities to handle post-comparison queries.
