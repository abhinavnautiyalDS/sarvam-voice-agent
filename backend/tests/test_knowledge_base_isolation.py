"""Regression tests for strict per-agent Knowledge Base isolation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend import knowledge_base


class KnowledgeBaseIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_root = knowledge_base.DATA_ROOT
        knowledge_base.DATA_ROOT = Path(self.temp_dir.name)

    def tearDown(self):
        knowledge_base.DATA_ROOT = self.original_root
        self.temp_dir.cleanup()

    def test_agent_databases_and_indexes_are_separate(self):
        def fake_embed(texts):
            return np.ones((len(texts), knowledge_base.DIMENSIONS), dtype="float32")

        with patch.object(knowledge_base, "_embed", side_effect=fake_embed):
            knowledge_base.ingest_text("general", "General only", "General private content", "general.md")
            knowledge_base.ingest_text("tutor", "Tutor only", "Tutor private content", "tutor.md")

            general_docs = knowledge_base.list_documents("general")
            tutor_docs = knowledge_base.list_documents("tutor")

            self.assertEqual([item["title"] for item in general_docs], ["General only"])
            self.assertEqual([item["title"] for item in tutor_docs], ["Tutor only"])

            general_db, general_index = knowledge_base._paths("general")
            tutor_db, tutor_index = knowledge_base._paths("tutor")
            self.assertNotEqual(general_db, tutor_db)
            self.assertNotEqual(general_index, tutor_index)
            self.assertTrue(general_db.exists())
            self.assertTrue(tutor_db.exists())
            self.assertTrue(general_index.exists())
            self.assertTrue(tutor_index.exists())

    def test_retrieval_enablement_is_agent_local(self):
        with patch.object(
            knowledge_base, "_embed",
            side_effect=lambda texts: np.ones((len(texts), knowledge_base.DIMENSIONS), dtype="float32"),
        ):
            knowledge_base.ingest_text("general", "General", "General source passage", "general.md")
            knowledge_base.ingest_text("customer_service", "Admissions", "Admissions source passage", "admissions.md")
            knowledge_base.set_enabled("general", True)

            self.assertTrue(knowledge_base.status("general")["enabled"])
            self.assertFalse(knowledge_base.status("customer_service")["enabled"])
            self.assertEqual(knowledge_base.retrieve("customer_service", "admission question"), [])

            results = knowledge_base.retrieve("general", "general question")
            self.assertTrue(results)
            self.assertTrue(all(item["title"] == "General" for item in results))


if __name__ == "__main__":
    unittest.main()
