# GitHub Repo RAG Analyzer

## Overview

The GitHub Repo RAG Analyzer is a powerful tool that combines the capabilities of Retrieval-Augmented Generation (RAG) with code analysis. This application allows users to process GitHub repositories (both local and remote) and create a knowledge base from the code content. Users can then query this knowledge base to get insights and answers about the codebase.

## Features

- Process local and remote GitHub repositories
- Chunk and embed code using advanced NLP techniques
- Store code embeddings in Google Firestore for efficient retrieval
- Query the processed codebase using natural language
- Generate responses using Google's Gemini 1.5 Pro AI model
- User-friendly Streamlit interface

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- A Google Cloud account with Firestore and Vertex AI enabled
- Necessary Google Cloud credentials set up on your machine

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/github-repo-rag-analyzer.git
   cd github-repo-rag-analyzer
   ```

2. Install the required packages:
   ```
   pip install streamlit gitpython google-cloud-firestore google-cloud-aiplatform langchain langchain-google-vertexai
   ```

3. Set up your Google Cloud project:
   - Create a new project or select an existing one
   - Enable Firestore in Native mode
   - Enable Vertex AI API
   - Set up authentication (download your service account key and set the GOOGLE_APPLICATION_CREDENTIALS environment variable)

4. Update the `PROJECT_ID` in the script with your Google Cloud project ID.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. In the web interface:
   - Choose between processing a local or remote repository
   - For local repos, enter the path to your repository
   - For remote repos, enter the GitHub URL and a local directory to clone into
   - Click "Process Repository" to analyze the codebase
   - Once processed, enter queries in natural language to get information about the code

## How it Works

1. **Repository Processing**: The tool clones (if remote) and processes the repository, reading all Python files.
2. **Code Chunking**: It splits the code into meaningful chunks using LangChain's RecursiveCharacterTextSplitter.
3. **Embedding Generation**: Each code chunk is converted into a vector embedding using Google's TextEmbedding model.
4. **Storage**: The chunks and their embeddings are stored in Firestore, creating a searchable knowledge base.
5. **Querying**: User queries are converted to embeddings and used to find the most relevant code chunks in Firestore.
6. **Response Generation**: Retrieved code chunks are used as context for Gemini 1.5 Pro to generate informative responses.

## Limitations

- Currently only processes Python files (.py extension)
- Performance may vary with very large repositories
- Requires a Google Cloud account and may incur costs

## Contributing

Contributions to improve the GitHub Repo RAG Analyzer are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Google Cloud for providing Firestore and Vertex AI services
- The LangChain community for their excellent tools and documentation