#!/usr/bin/env python3
"""
Document Loader Script for Genestory Chatbot
Loads documents from the knowledge base into the vector database.
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.embedding.manager import embedding_manager
from database.milvus_manager import milvus_manager
from config.settings import settings
from loguru import logger

# Configure logging
logger.add("logs/document_loader.log", rotation="10 MB")


def read_markdown_file(file_path: str) -> Dict[str, Any]:
    """Read and parse a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into sections (simple approach)
        sections = []
        current_section = ""
        current_title = ""
        
        for line in content.split('\n'):
            if line.startswith('#'):
                if current_section.strip():
                    sections.append({
                        "title": current_title,
                        "content": current_section.strip(),
                        "source": os.path.basename(file_path),
                        "section_type": "content"
                    })
                current_title = line.strip()
                current_section = ""
            else:
                current_section += line + "\n"
        
        # Add the last section
        if current_section.strip():
            sections.append({
                "title": current_title,
                "content": current_section.strip(),
                "source": os.path.basename(file_path),
                "section_type": "content"
            })
        
        return sections
        
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return []


def read_text_file(file_path: str) -> Dict[str, Any]:
    """Read a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return [{
            "title": os.path.basename(file_path),
            "content": content,
            "source": os.path.basename(file_path),
            "section_type": "document"
        }]
        
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return []


def chunk_text(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for sentence ending
            for i in range(end, start + max_chunk_size - 200, -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
            # Look for paragraph break
            else:
                for i in range(end, start + max_chunk_size - 200, -1):
                    if text[i] == '\n':
                        end = i + 1
                        break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


async def load_documents_from_directory(directory: str) -> List[Dict[str, Any]]:
    """Load all documents from a directory."""
    documents = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.warning(f"Directory {directory} does not exist")
        return documents
    
    logger.info(f"Loading documents from {directory}")
    
    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.md':
                sections = read_markdown_file(str(file_path))
                documents.extend(sections)
            elif file_ext in ['.txt', '.text']:
                sections = read_text_file(str(file_path))
                documents.extend(sections)
            else:
                logger.info(f"Skipping unsupported file type: {file_path}")
    
    logger.info(f"Loaded {len(documents)} document sections")
    return documents


async def process_and_upload_documents(documents: List[Dict[str, Any]]) -> bool:
    """Process documents and upload to vector database."""
    try:
        logger.info("Processing and uploading documents...")
        
        # Ensure models are loaded
        if not embedding_manager.is_loaded():
            logger.info("Loading embedding model...")
            await embedding_manager.load_model()
        
        # Connect to database
        if not milvus_manager.is_connected:
            logger.info("Connecting to Milvus database...")
            await milvus_manager.connect()
            await milvus_manager.create_collection()
        
        # Process documents in batches
        batch_size = 10
        processed_count = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Prepare batch data
            batch_contents = []
            batch_document_ids = []
            batch_sources = []
            batch_metadata = []
            
            for j, doc in enumerate(batch):
                # Split long content into chunks
                content = doc["content"]
                if len(content) > 1000:
                    chunks = chunk_text(content)
                else:
                    chunks = [content]
                
                for k, chunk in enumerate(chunks):
                    batch_contents.append(chunk)
                    batch_document_ids.append(f"{doc['source']}_{i+j}_{k}")
                    batch_sources.append(doc["source"])
                    
                    metadata = {
                        "title": doc.get("title", ""),
                        "section_type": doc.get("section_type", "content"),
                        "chunk_index": k,
                        "total_chunks": len(chunks)
                    }
                    batch_metadata.append(json.dumps(metadata))
            
            # Generate embeddings
            embeddings = embedding_manager.encode_documents(batch_contents)
            
            # Upload to database
            success = await milvus_manager.insert_documents(
                documents=batch_contents,
                embeddings=embeddings,
                document_ids=batch_document_ids,
                sources=batch_sources,
                metadata=batch_metadata
            )
            
            if success:
                processed_count += len(batch_contents)
                logger.info(f"Uploaded batch {i//batch_size + 1}, total processed: {processed_count}")
            else:
                logger.error(f"Failed to upload batch {i//batch_size + 1}")
                return False
        
        logger.info(f"Successfully processed and uploaded {processed_count} document chunks")
        return True
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        return False


async def main():
    """Main function to load documents."""
    try:
        logger.info("Starting document loading process...")
        
        # Load documents from sample data directory
        sample_data_dir = os.path.join(settings.KNOWLEDGE_BASE_DIR, "sample_data")
        documents = await load_documents_from_directory(sample_data_dir)
        
        if not documents:
            logger.warning("No documents found to load")
            return
        
        # Process and upload documents
        success = await process_and_upload_documents(documents)
        
        if success:
            logger.info("Document loading completed successfully")
            
            # Get final statistics
            stats = await milvus_manager.get_collection_stats()
            logger.info(f"Total documents in database: {stats.get('num_entities', 0)}")
        else:
            logger.error("Document loading failed")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        # Cleanup
        if milvus_manager.is_connected:
            await milvus_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
