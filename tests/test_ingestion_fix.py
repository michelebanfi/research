
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os
import asyncio

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to mock streamlit before importing ingest, as it runs code on import
sys.modules['streamlit'] = MagicMock()

from src.ui.ingest import process_upload
from src.ingestion import FileType
from src.database import DatabaseClient
from src.ai_engine import AIEngine

class TestIngestionFix(unittest.TestCase):
    def test_process_upload_keeps_failed_embeddings(self):
        # Mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "test.md"
        mock_file.getvalue.return_value = b"# Heading\n\nContent"
        
        # Mock DB
        mock_db = MagicMock(spec=DatabaseClient)
        mock_db.upload_file_metadata.return_value = {'id': 'test-file-id'}
        mock_db.store_sections.return_value = {}
        
        # Mock AI
        mock_ai = MagicMock(spec=AIEngine)
        mock_ai.embed_model = "test-model"
        mock_ai.generate_summary_async = AsyncMock(return_value="Summary")
        mock_ai.extract_metadata_graph_async = AsyncMock(return_value={"nodes": [], "edges": []})
        
        # Mock AsyncClient
        with patch('ollama.AsyncClient') as mock_client_cls:
            mock_client = mock_client_cls.return_value
            
            # Setup embeddings mock to return exception for parent, success for child
            async def embedding_side_effect(model, prompt):
                if "parent" in prompt:
                    raise Exception("Simulated Failure")
                return {"embedding": [0.1, 0.2, 0.3]}
            
            mock_client.embeddings = AsyncMock(side_effect=embedding_side_effect)
            
            # Mock dependencies
            with patch('src.ui.ingest.FileRouter.route') as mock_route, \
                 patch('src.ui.ingest.DoclingParser') as mock_parser_cls, \
                 patch('src.utils.async_utils.run_sync') as mock_run_sync:
                
                mock_route.return_value = FileType.DOCUMENT
                
                # Mock Parser Chunks
                mock_parser = mock_parser_cls.return_value
                chunks = [
                    {
                        "id": "parent-id", 
                        "content": "parent content",
                        "chunk_level": 0,
                        "parent_chunk_id": None,
                        "metadata": {"headings": ["Heading"]},
                        "embedding_text": "parent content" 
                    },
                    {
                        "id": "child-id",
                        "content": "child content",
                        "chunk_level": 1, 
                        "parent_chunk_id": "parent-id",
                        "metadata": {"headings": ["Heading"]},
                        "embedding_text": "child content"
                    }
                ]
                mock_parser.parse.return_value = chunks
                
                # Mock run_sync to execute the coroutine synchronously
                def fake_run_sync(coro):
                     return asyncio.run(coro)
                mock_run_sync.side_effect = fake_run_sync
                
                # EXECUTE
                process_upload(mock_file, mock_db, mock_ai)
                
                # VERIFY
                # Verify store_chunks called
                self.assertTrue(mock_db.store_chunks.called)
                
                # Get stored chunks
                stored_chunks = mock_db.store_chunks.call_args[0][1]
                
                # Check count
                print(f"Stored chunks count: {len(stored_chunks)}")
                self.assertEqual(len(stored_chunks), 2)
                
                # Check Parent (Failed Embedding -> None)
                parent = next((c for c in stored_chunks if c['id'] == 'parent-id'), None)
                self.assertIsNotNone(parent)
                self.assertIsNone(parent.get('embedding'))
                print("Parent chunk preserved with None embedding.")
                
                # Check Child (Success -> List)
                child = next((c for c in stored_chunks if c['id'] == 'child-id'), None)
                self.assertIsNotNone(child)
                self.assertEqual(child.get('embedding'), [0.1, 0.2, 0.3])
                print("Child chunk preserved with valid embedding.")

if __name__ == '__main__':
    unittest.main()
