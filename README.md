# TCC-Helper 1.0

TCC-Helper is a Python-based command-line interface (CLI) tool designed to help you work with powerful AI models locally. It integrates seamlessly with Ollama's local server and llama.cpp, allowing you to run AI models efficiently on your machine without relying on cloud services. Whether you're querying data, populating a database, or testing RAG (Retrieval-Augmented Generation), TCC-Helper provides a lightweight and customizable solution for your AI needs.
Features

    Local Server Integration: Run AI models locally using Ollama or llama.cpp.

    Python-Based: Easy to use and integrate into your Python workflows.

    Customizable: Interact with AI models in a way that fits your specific requirements.

    Efficient: Lightweight and fast, optimized for performance.

    RAG Support: Work with Retrieval-Augmented Generation for enhanced querying capabilities.

    No GUI: A CLI-focused tool for quick and efficient workflows.

## Updates
## Version 1.0

    Added support for llama.cpp alongside Ollama.

    Improved performance and stability.

    Enhanced RAG functionality for better querying and document retrieval.

    Simplified CLI commands for ease of use.

## Installation

Clone the Repository:
    
    git clone https://github.com/marcosdotonion/tcc-helper.git
    cd tcc-helper

Set Up a Python Virtual Environment:
    
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install Dependencies:
    
    pip install -r requirements.txt

Set Up Ollama or llama.cpp:

    For Ollama, ensure the local server is running. Follow the Ollama documentation for setup instructions.

    For llama.cpp, ensure it is installed and configured on your system. Refer to the llama.cpp GitHub repository for setup instructions.

## Usage Commands

Query Data:
Query the AI model with a specific prompt.

    python3 query_data.py "PROMPT"
or:

    python3 query_cpp.py
    > Enter your prompt

Populate Database:
Add documents to the RAG database for enhanced querying.

    python3 populate_database.py
or:

    python3 populate_cpp.py

Test RAG (Work in Progress):
Test the Retrieval-Augmented Generation functionality.

    python3 test_rag.py

## Configuration
## Environment Variables

    OLLAMA_URL: Set the URL for the Ollama server (default: http://localhost:11434).

    LLAMA_CPP_URL: Set the URL for the llama.cpp server (default: http://localhost:8080).

## Example Configuration (not needed, useful)

Create a .env file in the root directory:

    OLLAMA_URL=http://localhost:11434
    LLAMA_CPP_URL=http://localhost:8080

## Contributing

Contributions are welcome! If you'd like to contribute to TCC-Helper, please follow these steps:

    Fork the repository.

    Create a new branch for your feature or bugfix.

    Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Ollama For providing a powerful local AI server.

llama.cpp For enabling efficient AI model inference on local machines.

LangChain For simplifying the integration of AI models and RAG functionality.

## Support

For questions, issues, or feature requests, please open an issue on the GitHub repository.

Enjoy using TCC-Helper 1.0! 🚀
