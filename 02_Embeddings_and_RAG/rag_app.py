import chainlit as cl
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter, PDFFileLoader
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
import asyncio
import tempfile
import os

# Initialize the RAG components
text_loader = TextFileLoader("data/PMarcaBlogs.txt")
initial_documents = text_loader.load_documents()

text_splitter = CharacterTextSplitter()
text_chunks = text_splitter.split_texts(initial_documents)

# Create vector database
vector_db = VectorDatabase()
vector_db = asyncio.run(vector_db.abuild_from_list(text_chunks))

# Initialize OpenAI chat model
chat_openai = ChatOpenAI()

# Define RAG prompt templates
RAG_PROMPT_TEMPLATE = """ \
You are a helpful assistant that answers questions based on the provided context.

Use the provided context to answer the user's query. If the context contains relevant information, use it to provide a detailed answer.

If the context is in another language, translate it to English in your response.

If the context doesn't contain enough information to fully answer the query, acknowledge what you know from the context and indicate what information is missing.

Only respond with "I don't know" if the context contains absolutely no relevant information about the query.
"""

rag_prompt = SystemRolePrompt(RAG_PROMPT_TEMPLATE)

USER_PROMPT_TEMPLATE = """ \
Context:
{context}

User Query:
{user_query}
"""

user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    def run_pipeline(self, user_query: str) -> str:
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)

        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_system_prompt = rag_prompt.create_message()
        formatted_user_prompt = user_prompt.create_message(user_query=user_query, context=context_prompt)

        return {"response": self.llm.run([formatted_system_prompt, formatted_user_prompt]), "context": context_list}
    
    def add_document(self, file_path: str) -> None:
        """Add a new document to the vector database."""
        if file_path.endswith('.pdf'):
            loader = PDFFileLoader(file_path)
        else:
            loader = TextFileLoader(file_path)
            
        new_documents = loader.load_documents()
        new_chunks = text_splitter.split_texts(new_documents)
        
        # Add new chunks to vector database
        asyncio.run(self.vector_db_retriever.abuild_from_list(new_chunks))

# Initialize the RAG pipeline
rag_pipeline = RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=chat_openai
)

@cl.on_chat_start
async def start():
    # Configure file upload settings
    await cl.Message(
        content="Hello! I'm your RAG-powered assistant. You can upload PDF or text files and ask me anything about them!",
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Check if there are any uploaded files
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                # Create a temporary file to store the uploaded content
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(element.name)[1]) as temp_file:
                    # Write the uploaded file content to the temporary file
                    temp_file.write(element.content)
                    temp_file_path = temp_file.name
                
                try:
                    # Add the document to the RAG pipeline
                    rag_pipeline.add_document(temp_file_path)
                    await cl.Message(
                        content=f"Successfully processed {element.name}! You can now ask questions about it.",
                    ).send()
                except Exception as e:
                    await cl.Message(
                        content=f"Error processing {element.name}: {str(e)}",
                    ).send()
                finally:
                    # Clean up the temporary file
                    os.unlink(temp_file_path)
                return
    
    # If no files were uploaded, process the query
    result = rag_pipeline.run_pipeline(message.content)
    
    # Create response message
    response = cl.Message(content=result["response"])
    
    # Add context as elements
    elements = []
    for context, score in result["context"]:
        elements.append(
            cl.Text(
                name=f"Context (Score: {score:.4f})",
                content=context,
                display="inline"
            )
        )
    
    response.elements = elements
    await response.send()