import ollama
import chromadb
import logging
import pandas as pd
from typing import List, Dict, Tuple
import os
from tqdm import tqdm
from chromadb.config import Settings
import datetime

# Configure logging
logger = logging.getLogger(__name__)

class ClipboardEmbeddings:
    def __init__(self, embedding_model: str = "all-minilm", chunk_size: int = 1000, chunk_overlap: int = 200,
                 force_recreate: bool = False):
        """Initialize the embeddings handler with specified model."""
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create a persistent client with a specific path
        persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        if not force_recreate:
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name="clipboard_embeddings"
                )
                logger.info("Using existing collection")
            except Exception as e:
                logger.info("Creating new collection")
                self.collection = self.chroma_client.create_collection(
                    name="clipboard_embeddings",
                    metadata={"description": "Embeddings for clipboard content"}
                )
        else:
            logger.info("Recreating collection")
            try:
                self.chroma_client.delete_collection(name="clipboard_embeddings")
                logger.info("Deleted existing collection")
            except Exception as e:
                logger.info("No existing collection to delete")
            self.collection = self.chroma_client.create_collection(
                name="clipboard_embeddings",
                metadata={"description": "Embeddings for clipboard content"},

            )

        logger.info(f"Initialized embeddings handler with model: {embedding_model}")

    def initialize_embeddings(self, df: pd.DataFrame) -> bool:
        """Initialize embeddings for the provided DataFrame."""
        if df is None or len(df) == 0:
            logger.warning("No data provided for initialization")
            return False

        try:
            # Prepare all items
            items = [
                {
                    'id': idx,
                    'content': row['item']
                }
                for idx, row in df.iterrows()
            ]
            
            # Process in batches
            batch_size = 100
            total_batches = (len(items) + batch_size - 1) // batch_size
            logger.info(f"Processing {len(items)} items in {total_batches} batches")
            
            success = self.add_batch(items, batch_size)
            
            if success:
                logger.info(f"Successfully initialized embeddings for {len(items)} items")
            else:
                logger.error("Some items failed during initialization")
            
            return success
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            return False

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for a single text item."""
        try:
            # Ensure text is a string and not too long
            text = str(text)  # Increased length limit
            response = ollama.embed(
                model=self.embedding_model,
                input=text
            )
            # The response contains a single embedding vector
            if "embeddings" not in response:  # Note: changed from "embeddings" to "embedding"
                logger.error(f"Unexpected response format: {response}")
                raise ValueError("No embedding in response")
            return response["embeddings"][0]  # This is already a list of floats
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def create_chunks(self, text: str, doc_id: str) -> List[Tuple[str, str, Dict]]:
        """Create overlapping chunks from text with document reference."""
        chunks = []
        text = str(text)  # Ensure text is string
        
        if len(text) <= self.chunk_size:
            # If text is shorter than chunk size, keep it as one chunk
            chunks.append((
                f"{doc_id}-0",  # chunk_id
                text,
                {"original_id": doc_id, "chunk_index": 0, "total_chunks": 1}
            ))
            return chunks
        
        start = 0
        chunk_index = 0
        
        while start+self.chunk_overlap < len(text):
            # Calculate end position for current chunk
            end = min(start + self.chunk_size, len(text))
            
            # Get the chunk
            chunk = text[start:end]
            
            # Add to chunks list
            chunks.append((
                f"{doc_id}-{chunk_index}",  # chunk_id
                chunk,
                {"original_id": doc_id, "chunk_index": chunk_index}
            ))
            
            # Move start position for next chunk, considering overlap
            start = end - self.chunk_overlap
            chunk_index += 1
            
            # Break if we've reached the end
            if start >= len(text):
                break
        
        # Update metadata with total chunks
        chunks = [(id, text, {**meta, "total_chunks": len(chunks)}) 
                 for id, text, meta in chunks]
        
        return chunks

    def add_batch(self, items: List[Dict[str, str]], batch_size: int = 10) -> bool:
        """Add a batch of items to the vector store."""
        try:
            # First create all chunks for all documents
            logger.info("Starting chunking process...")
            start_time = datetime.datetime.now()
            all_chunks = []
            for i, item in enumerate(items):
                if i % 10 == 0:  # Log every 10 items
                    logger.info(f"Chunking item {i+1}/{len(items)}")
                chunks = self.create_chunks(item['content'], str(item['id']))
                all_chunks.extend(chunks)
            
            chunk_time = datetime.datetime.now() - start_time
            logger.info(f"Chunking complete. Created {len(all_chunks)} chunks from {len(items)} items in {chunk_time.total_seconds():.2f}s")
            
            # Process chunks in batches
            logger.info(f"Starting embedding generation for {len(all_chunks)} chunks...")
            embed_start_time = datetime.datetime.now()
            successful_chunks = 0
            
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
                batch_start_time = datetime.datetime.now()
                
                embeddings = []
                ids = []
                documents = []
                metadatas = []
                
                for chunk_id, chunk_text, chunk_metadata in batch:
                    try:
                        embedding = self.generate_embedding(chunk_text)
                        if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
                            raise ValueError(f"Invalid embedding format for chunk {chunk_id}")
                        embeddings.append(embedding)
                        ids.append(chunk_id)
                        documents.append(chunk_text)
                        metadatas.append(chunk_metadata)
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                        continue
                
                if embeddings:
                    try:
                        self.collection.add(
                            ids=ids,
                            embeddings=embeddings,
                            documents=documents,
                            metadatas=metadatas
                        )
                        successful_chunks += len(embeddings)
                        batch_time = datetime.datetime.now() - batch_start_time
                        logger.info(f"Added batch of {len(embeddings)} chunks. Batch took {batch_time.total_seconds():.2f}s")
                    except Exception as e:
                        logger.error(f"Error adding batch to collection: {str(e)}")
                        logger.error(f"First embedding shape: {len(embeddings[0]) if embeddings else 'No embeddings'}")
            
            total_time = datetime.datetime.now() - start_time
            logger.info(f"Embedding complete. Added {successful_chunks}/{len(all_chunks)} chunks in {total_time.total_seconds():.2f}s")
            
            return True
        except Exception as e:
            logger.error(f"Error in add_batch: {str(e)}")
            return False

    def find_similar(self, query: str, n_results: int = 5) -> Dict:
        """Find similar items to the query text."""
        try:
            query_embedding = self.generate_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count())
            )
            return results
        except Exception as e:
            logger.error(f"Error finding similar items: {str(e)}")
            raise

    def get_collection_info(self) -> Dict:
        """Get information about the current collection."""
        try:
            return {
                "count": self.collection.count(),
                "name": self.collection.name,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise
