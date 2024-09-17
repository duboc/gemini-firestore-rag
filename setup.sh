#!/bin/bash

# GitHub Repo RAG Analyzer Setup Script

echo "Welcome to the GitHub Repo RAG Analyzer setup script!"
echo "This script will guide you through the setup process."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null
then
    echo "gcloud could not be found. Please install the Google Cloud SDK first."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Ensure the user is logged in
echo "Ensuring you're logged into gcloud..."
gcloud auth login

# Set the project ID
echo "Please enter your Google Cloud project ID:"
read PROJECT_ID
gcloud config set project $PROJECT_ID

# Enable necessary APIs
echo "Enabling necessary Google Cloud APIs..."
gcloud services enable firestore.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Set up Firestore
echo "Setting up Firestore in Native mode..."
gcloud app create --region=us-central
gcloud firestore databases create --region=us-central

# Create Firestore index
echo "Creating Firestore index for vector search..."
gcloud firestore indexes composite create \
--collection-group=github-code \
--query-scope=COLLECTION \
--field-config field-path=embedding,vector-config='{"dimension":"768", "flat": "{}"}' \
--project=$PROJECT_ID

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
echo "Installing required Python packages..."
pip install streamlit gitpython google-cloud-firestore google-cloud-aiplatform langchain langchain-google-vertexai

# Set up application default credentials
echo "Setting up application default credentials..."
gcloud auth application-default login

# Clone the repository (uncomment and modify if you want to include this step)
# echo "Cloning the GitHub Repo RAG Analyzer repository..."
# git clone https://github.com/yourusername/github-repo-rag-analyzer.git
# cd github-repo-rag-analyzer

echo "Setup complete! You can now run the analyzer with: streamlit run app.py"
echo "Make sure to update the PROJECT_ID in the app.py file with: $PROJECT_ID"