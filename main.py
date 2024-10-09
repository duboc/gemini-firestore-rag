import streamlit as st
import os
import git
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, ChatSession
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
import logging
import time
import concurrent.futures
import re
import json
from functools import wraps
from typing import List, Tuple, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StreamlitHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs: List[str] = []

    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
        except Exception:
            self.handleError(record)


streamlit_handler = StreamlitHandler()
streamlit_handler.setLevel(logging.INFO)
logger.addHandler(streamlit_handler)


# Configuration
class Config:
    PROJECT_ID: str = "jose-genai-demos"
    LOCATION: str = "us-central1"
    EMBEDDING_MODEL: str = "textembedding-gecko"
    GENERATIVE_MODEL: str = "gemini-1.5-pro-001"
    COLLECTION_NAME: str = "github-code"


class CostConfig:
    FIRESTORE_FREE_READS: int = 50000
    FIRESTORE_FREE_WRITES: int = 20000
    FIRESTORE_READ_COST: float = 0.03 / 100000
    FIRESTORE_WRITE_COST: float = 0.09 / 100000
    FIRESTORE_STORAGE_COST: float = 0.15 / 1024  # per GiB/month
    VERTEX_EMBEDDING_COST: float = 0.000025 / 1000  # per 1000 characters
    VERTEX_GEMINI_INPUT_COST: float = 0.00125 / 1000  # per 1000 characters
    VERTEX_GEMINI_OUTPUT_COST: float = 0.00375 / 1000  # per 1000 characters


# Initialize services
vertexai.init(project=Config.PROJECT_ID, location=Config.LOCATION)
db = firestore.Client(project=Config.PROJECT_ID)
embedding_model = VertexAIEmbeddings(model_name=Config.EMBEDDING_MODEL)
gemini_pro = GenerativeModel(Config.GENERATIVE_MODEL)


def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"An error occurred in {func.__name__}: {str(e)}")

    return wrapper


@st.cache_resource
def get_chat_session() -> ChatSession:
    return gemini_pro.start_chat(history=[])


@error_handler
def clone_or_pull_repo(repo_url: str, local_path: str = "repo") -> None:
    logger.info(f"Checking repository at {local_path}")
    if os.path.exists(local_path):
        logger.info(f"Repository exists at {local_path}. Pulling latest changes.")
        repo = git.Repo(local_path)
        origin = repo.remotes.origin
        origin.pull()
    else:
        logger.info(f"Cloning repository from {repo_url} to {local_path}")
        git.Repo.clone_from(repo_url, local_path)
    logger.info(f"Repository ready at {local_path}")


@error_handler
def process_local_repo(repo_path: str) -> List[Tuple[str, str]]:
    logger.info(f"Processing local repository at {repo_path}")
    processed_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    logger.info(f"Processed file: {file_path}")
                    processed_files.append((file_path, content))
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
    return processed_files


@error_handler
def chunk_and_embed_code(content: str) -> Tuple[List[str], List[List[float]]]:
    logger.info("Chunking and embedding code")
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_text(content)
    logger.info(f"Created {len(chunks)} chunks")
    embeddings = embedding_model.embed_documents(chunks)
    logger.info(f"Generated {len(embeddings)} embeddings")
    return chunks, embeddings


@error_handler
def store_in_firestore(chunks: List[str], embeddings: List[List[float]]) -> None:
    logger.info(f"Checking if '{Config.COLLECTION_NAME}' collection exists")
    collection = db.collection(Config.COLLECTION_NAME)
    if not collection.get():
        logger.info(
            f"Collection '{Config.COLLECTION_NAME}' does not exist. Creating and populating it."
        )
        batch = db.batch()
        for i, (content, embedding) in enumerate(zip(chunks, embeddings)):
            doc_ref = collection.document(f"doc_{i}")
            batch.set(doc_ref, {"content": content, "embedding": Vector(embedding)})
        batch.commit()
        logger.info(f"Stored {len(chunks)} documents in Firestore")
    else:
        logger.info(
            f"Collection '{Config.COLLECTION_NAME}' already exists. Skipping population."
        )


@error_handler
def search_vector_database(query: str) -> List[str]:
    logger.info(f"Searching vector database for query: {query}")
    collection = db.collection(Config.COLLECTION_NAME)
    query_embedding = embedding_model.embed_query(query)
    vector_query = collection.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=5,
    )
    docs = vector_query.stream()
    context = [result.to_dict()["content"] for result in docs]
    logger.info(f"Found {len(context)} relevant documents")
    return context


@error_handler
def generate_code(chat: ChatSession, query: str, context: str) -> str:
    logger.info(f"Generating code for query: {query}")
    prompt = f"""You are an expert Python developer with deep knowledge of software architecture and best practices. 
    Your task is to generate high-quality, efficient, and well-documented Python code based on the following query and context.
    Please ensure the code is:
    1. Modular and follows SOLID principles
    2. Well-commented and easy to understand
    3. Efficient and optimized for performance
    4. Handles potential errors and edge cases
    5. Follows PEP 8 style guidelines
    6. Do NOT suggest incomplete code such as comments to the user implement with their own code

    Query: {query}

    Context (relevant code from the repository):
    {context}

    Please generate the requested code, explaining your design decisions and any assumptions you've made."""

    logger.info(json.dumps({"request": prompt}))
    response = chat.send_message(prompt)
    logger.info(json.dumps({"response": response.text}))
    return response.text


def extract_python_code(text: str) -> str:
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    return "\n\n".join(code_block.strip() for code_block in code_blocks)


def count_tokens(model: GenerativeModel, text: str) -> Tuple[int, int]:
    response = model.count_tokens(text)
    return response.total_tokens, response.total_billable_characters


def calculate_firestore_cost(
    reads: int, writes: int, storage_gb: float
) -> Dict[str, float]:
    read_cost = (
        max(0, reads - CostConfig.FIRESTORE_FREE_READS) * CostConfig.FIRESTORE_READ_COST
    )
    write_cost = (
        max(0, writes - CostConfig.FIRESTORE_FREE_WRITES)
        * CostConfig.FIRESTORE_WRITE_COST
    )
    storage_cost = storage_gb * CostConfig.FIRESTORE_STORAGE_COST
    return {
        "read_cost": read_cost,
        "write_cost": write_cost,
        "storage_cost": storage_cost,
        "total_cost": read_cost + write_cost + storage_cost,
    }


def calculate_vertex_cost(
    input_chars: int, output_chars: int, embedding_chars: int
) -> Dict[str, float]:
    embedding_cost = embedding_chars * CostConfig.VERTEX_EMBEDDING_COST / 1000
    input_cost = input_chars * CostConfig.VERTEX_GEMINI_INPUT_COST / 1000
    output_cost = output_chars * CostConfig.VERTEX_GEMINI_OUTPUT_COST / 1000
    return {
        "embedding_cost": embedding_cost,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": embedding_cost + input_cost + output_cost,
    }


def estimate_costs(
    repo_size: float, num_files: int, num_queries: int
) -> Dict[str, Any]:
    # Estimating Firestore usage
    estimated_chunks = num_files * 10  # Assuming average 10 chunks per file
    estimated_reads = num_queries * 5  # Assuming 5 reads per query for context
    estimated_writes = estimated_chunks
    estimated_storage = repo_size * 1.5  # Assuming 1.5x storage due to indexing

    firestore_costs = calculate_firestore_cost(
        estimated_reads, estimated_writes, estimated_storage
    )

    # Estimating Vertex AI usage
    avg_chunk_size = 1000  # characters
    avg_query_size = 200  # characters
    avg_response_size = 1000  # characters

    embedding_chars = estimated_chunks * avg_chunk_size
    input_chars = (num_queries * avg_query_size) + (estimated_reads * avg_chunk_size)
    output_chars = num_queries * avg_response_size

    vertex_costs = calculate_vertex_cost(input_chars, output_chars, embedding_chars)

    total_cost = firestore_costs["total_cost"] + vertex_costs["total_cost"]

    return {
        "firestore": firestore_costs,
        "vertex": vertex_costs,
        "total_cost": total_cost,
    }


def display_log():
    st.subheader("Log")
    if streamlit_handler.logs:
        for log in streamlit_handler.logs:
            try:
                log_data = json.loads(log)
                if "request" in log_data or "response" in log_data:
                    st.json(log_data)
            except json.JSONDecodeError:
                st.text(log)
    else:
        st.info("No logs to display yet.")


def process_repository(repo_type: str, repo_path: str, repo_url: str) -> None:
    with st.spinner("Processing repository..."):
        progress_bar = st.progress(0)
        if repo_type == "Remote":
            clone_or_pull_repo(repo_url)
            repo_path = "repo"

        all_chunks = []
        all_embeddings = []
        processed_files = process_local_repo(repo_path)
        for i, (file_path, content) in enumerate(processed_files):
            chunks, embeddings = chunk_and_embed_code(content)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
            progress_bar.progress(min((i + 1) / len(processed_files), 1.0))

        store_in_firestore(all_chunks, all_embeddings)
        progress_bar.progress(1.0)

    st.success(
        f"Repository processed and stored {len(all_chunks)} chunks in Firestore!"
    )


def generate_code_sequence(initial_query: str, num_iterations: int) -> None:
    with st.spinner("Generating code sequence..."):
        chat = get_chat_session()
        context = search_vector_database(initial_query)

        st.session_state.generated_code = []
        st.session_state.queries = [initial_query]
        st.session_state.rag_references = [context]

        for i in range(num_iterations):
            logger.info(f"Generating code for iteration {i+1}")
            code = generate_code(
                chat,
                st.session_state.queries[-1],
                "\n\n".join(st.session_state.rag_references[-1]),
            )
            st.session_state.generated_code.append(code)

            if i < num_iterations - 1:
                followup_prompt = f"Based on the previous code, suggest a new query that would improve or extend the functionality. Only provide the query, not the code."
                logger.info(json.dumps({"request": followup_prompt}))
                new_query = chat.send_message(followup_prompt).text
                logger.info(json.dumps({"response": new_query}))
                st.session_state.queries.append(new_query)
                new_context = search_vector_database(new_query)
                st.session_state.rag_references.append(new_context)

    st.success(
        "Code generation sequence complete! Switch to the 'Generated Code' tab to view the results."
    )


def display_generated_code():
    st.subheader("Generated Code")
    if "generated_code" in st.session_state:
        for i, (code, query, refs) in enumerate(
            zip(
                st.session_state.generated_code,
                st.session_state.queries,
                st.session_state.rag_references,
            ),
            1,
        ):
            st.subheader(f"Code Version {i} - Query: {query}")

            with st.expander("Generated Code", expanded=True):
                python_code = extract_python_code(code)
                st.code(python_code, language="python")
                if st.button(f"Explain Code {i}"):
                    chat = get_chat_session()
                    explanation_prompt = (
                        f"Explain the following Python code in detail:\n\n{python_code}"
                    )
                    logger.info(json.dumps({"request": explanation_prompt}))
                    explanation = chat.send_message(explanation_prompt).text
                    logger.info(json.dumps({"response": explanation}))
                    st.markdown(explanation)

            with st.expander("RAG References", expanded=False):
                for j, ref in enumerate(refs, 1):
                    st.text(f"Reference {j}:")
                    st.code(ref, language="python")


def display_costs():
    st.subheader("Cost Estimation")

    repo_size = st.number_input(
        "Estimated repository size (in MB)", min_value=1, value=100
    )
    num_files = st.number_input(
        "Estimated number of Python files", min_value=1, value=50
    )
    num_queries = st.number_input("Estimated number of queries", min_value=1, value=10)

    if st.button("Estimate Costs"):
        costs = estimate_costs(
            repo_size / 1024, num_files, num_queries
        )  # Convert MB to GB

        st.write("Firestore Costs:")
        st.write(f"- Read Cost: ${costs['firestore']['read_cost']:.4f}")
        st.write(f"- Write Cost: ${costs['firestore']['write_cost']:.4f}")
        st.write(f"- Storage Cost: ${costs['firestore']['storage_cost']:.4f}")
        st.write(f"- Total Firestore Cost: ${costs['firestore']['total_cost']:.4f}")

        st.write("\nVertex AI Costs:")
        st.write(f"- Embedding Cost: ${costs['vertex']['embedding_cost']:.4f}")
        st.write(f"- Input Cost: ${costs['vertex']['input_cost']:.4f}")
        st.write(f"- Output Cost: ${costs['vertex']['output_cost']:.4f}")
        st.write(f"- Total Vertex AI Cost: ${costs['vertex']['total_cost']:.4f}")

        st.write(f"\nTotal Estimated Cost: ${costs['total_cost']:.4f}")


def main():
    st.set_page_config(layout="wide")
    st.title("Advanced GitHub Repo RAG Analyzer and Code Generator")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Repository Processing",
            "Code Generation",
            "Generated Code",
            "Cost Estimation",
            "Log",
        ]
    )

    with tab1:
        repo_type = st.radio("Select repository type:", ("Local", "Remote"))
        repo_path = (
            st.text_input("Enter the path to your local repository:", value="repo")
            if repo_type == "Local"
            else None
        )
        repo_url = (
            st.text_input("Enter the URL of the GitHub repository:")
            if repo_type == "Remote"
            else None
        )

        if st.button("Process Repository"):
            process_repository(repo_type, repo_path, repo_url)

    with tab2:
        st.subheader("Code Generation")
        initial_query = st.text_area("Enter your initial code generation query:")
        num_iterations = st.slider(
            "Number of code generations:", min_value=1, max_value=5, value=3
        )

        if st.button("Generate Code Sequence"):
            generate_code_sequence(initial_query, num_iterations)

    with tab3:
        display_generated_code()

    with tab4:
        display_costs()

    with tab5:
        display_log()


if __name__ == "__main__":
    main()
