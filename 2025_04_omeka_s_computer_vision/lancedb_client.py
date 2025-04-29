import numpy as np
import lancedb
import pyarrow as pa
import logging
from dotenv import load_dotenv
import os
import ast

# Load env vars
load_dotenv(os.path.join(os.getcwd(), ".env"),override = True)
metadata_keys_raw = os.getenv("_DEFAULT_PARSE_METADATA", "").split(",")
metadata_keys = [key.replace(" ", "").replace(")", "").strip("'") for key in metadata_keys_raw]

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanceDBManager:

    def __init__(self, db_uri="lancedb", embedding_dim=768):
        self.db = lancedb.connect(db_uri)
        self.embedding_dim = embedding_dim
        logger.info(f"Connected to LanceDB at {db_uri}")

    def _build_schema(self):
        """Build LanceDB schema with dynamic metadata fields and embedding vector."""
        fields = [
            pa.field("id", pa.int64()),  
            pa.field("item_id", pa.string()),           
            pa.field("images_urls", pa.string()),
            pa.field("text", pa.string()),
            pa.field("Cluster", pa.string()),
            pa.field("Topic", pa.string()),
            pa.field("embeddings", pa.list_(pa.float32(), self.embedding_dim)),
            pa.field("umap_embeddings", pa.list_(pa.float32(), 2)),           
        ]

        # Add fields from metadata
        for key in metadata_keys:
            sanitized_key = key.split(":")[1].strip().capitalize() # remove the vocabulary prefix in key label and capitalize
            fields.append(pa.field(sanitized_key, pa.string()))

        return pa.schema(fields)

    def create_table(self, table_name):
        """Create table using dynamic schema."""
        try:
            schema = self._build_schema()
            table = self.db.create_table(table_name, schema=schema)
            logger.info(f"Created LanceDB table '{table_name}'")
            return table
        except Exception as e:
            logger.error(f"Failed to create table '{table_name}': {e}")
            raise

    def retrieve_table(self, table_name):
        try:
            table = self.db.open_table(table_name)
            logger.info(f"Opened existing LanceDB table '{table_name}'")
            return table
        except Exception as e:
            logger.error(f"Failed to open table '{table_name}': {e}")
            raise

    def initialize_table(self, table_name):
        try:
            return self.retrieve_table(table_name)
        except Exception:
            logger.info(f"Table '{table_name}' not found. Creating new.")
            return self.create_table(table_name)

    def add_entry(self, table_name, items):
        table = self.initialize_table(table_name)
        table.add(items)
        logger.info(f"Added items to table '{table_name}'")
        
    def list_tables(self):
        """List all existing tables in the LanceDB instance."""
        try:
            tables = self.db.table_names()
            logger.info("Retrieved list of tables.")
            return tables
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            raise
        
    def get_content_table(self, table_name):
        table = self.initialize_table(table_name)
        return table.to_pandas()
    
    def drop_table(self, table_name):
        """remove an existing table by name."""
        try:
            table = self.db.drop_table(table_name)
            logger.info(f"Remove existing LanceDB table '{table_name}' successfully.")
            return table
        except Exception as e:
            logger.error(f"Failed to remove existing table '{table_name}': {e}")
            raise
        
    def semantic_search(self, table_name, query_embed, limit):
        """
        Perform a semantic search using a provided text query or image.

        Args:
            query_text (str): The text query for the search.
            query_image_path (str): The path to the image for the search.
            limit (int): The maximum number of results to return.

        Returns:
            str: JSON string of search results.
        """
        table = self.initialize_table(table_name)
        #https://lancedb.github.io/lancedb/notebooks/DisappearingEmbeddingFunction/

        try:
            # Perform the search in LanceDB
            results = (table
                    .search(query_embed,vector_column_name="embeddings")
                    .distance_type("cosine")
                    .select(["id"])
                    .limit(limit)
                    .to_pandas()
                    #.sort_values(by='_distance', ascending=True)
                    .to_json(orient="records")
            )
            return results
        except Exception as e:
            raise
