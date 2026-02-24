"""
app/core/migrations.py
======================
Idempotent startup migrations applied at every app launch.

SQLAlchemy's ``create_all()`` creates *missing* tables but never alters
existing ones, so new columns and enum values added to ``domain.py``
accumulate as schema drift on any pre-existing database.  This module
fills that gap by applying every incremental DDL change with IF NOT EXISTS
/ IF EXISTS guards, making it safe to run against both fresh and
already-populated databases.

Call order in main.py:
  1. CREATE EXTENSION IF NOT EXISTS vector
  2. Base.metadata.create_all()   ← creates missing *tables* from current ORM
  3. run_startup_migrations()      ← adds missing *columns / enum values* to existing tables
"""
from __future__ import annotations

import logging
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def run_startup_migrations(engine: Engine) -> None:
    """Apply all incremental schema changes idempotently."""

    # ── 1. Enum values ────────────────────────────────────────────────────────
    # ALTER TYPE … ADD VALUE must be committed before subsequent DDL can
    # reference the new value.  We commit immediately after each one.
    enum_changes = [
        "ALTER TYPE filetype ADD VALUE IF NOT EXISTS 'TIFF'",
    ]
    with engine.connect() as conn:
        for stmt in enum_changes:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception as exc:
                conn.rollback()
                logger.warning("[migrations] enum change skipped (%s): %s", stmt, exc)

    # ── 2. Column additions ───────────────────────────────────────────────────
    # All statements use ADD COLUMN IF NOT EXISTS — safe on any Postgres 9.6+.
    column_stmts = [
        # raw_document — new audit / pipeline columns
        "ALTER TABLE raw_document ADD COLUMN IF NOT EXISTS created_at            TIMESTAMPTZ NOT NULL DEFAULT now()",
        "ALTER TABLE raw_document ADD COLUMN IF NOT EXISTS processed_at          TIMESTAMPTZ",
        "ALTER TABLE raw_document ADD COLUMN IF NOT EXISTS processing_directives JSONB",
        "ALTER TABLE raw_document ADD COLUMN IF NOT EXISTS processing_summary    JSONB",

        # parsed_layout_segment
        "ALTER TABLE parsed_layout_segment ADD COLUMN IF NOT EXISTS created_at   TIMESTAMPTZ NOT NULL DEFAULT now()",

        # semantic_child_chunk — chunk provenance metadata
        "ALTER TABLE semantic_child_chunk ADD COLUMN IF NOT EXISTS chunk_metadata JSONB",

        # extracted_entity — confidence, method, and canonical dedup FK
        "ALTER TABLE extracted_entity ADD COLUMN IF NOT EXISTS confidence         FLOAT",
        "ALTER TABLE extracted_entity ADD COLUMN IF NOT EXISTS extraction_method  VARCHAR",
        # canonical_id FK added separately below after ensuring the target table exists

        # knowledge_graph_edge — optional payload + audit timestamp
        # (column was originally named 'metadata' but that name is reserved by SQLAlchemy)
        "ALTER TABLE knowledge_graph_edge ADD COLUMN IF NOT EXISTS edge_metadata JSONB",
        "ALTER TABLE knowledge_graph_edge ADD COLUMN IF NOT EXISTS created_at     TIMESTAMPTZ NOT NULL DEFAULT now()",

        # retrieval_audit_log — per-stage latency breakdown + audit timestamp
        "ALTER TABLE retrieval_audit_log ADD COLUMN IF NOT EXISTS stage_latencies JSONB",
        "ALTER TABLE retrieval_audit_log ADD COLUMN IF NOT EXISTS created_at      TIMESTAMPTZ NOT NULL DEFAULT now()",
    ]

    with engine.connect() as conn:
        for stmt in column_stmts:
            try:
                conn.execute(text(stmt))
            except Exception as exc:
                logger.warning("[migrations] column stmt skipped (%s…): %s", stmt[:60], exc)
        conn.commit()

    # ── 3. canonical_id FK on extracted_entity ────────────────────────────────
    # This must run after create_all() has ensured canonical_entity exists.
    # We guard with a sub-select against information_schema to avoid duplicate FK.
    with engine.connect() as conn:
        try:
            conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE  table_name  = 'extracted_entity'
                        AND    column_name = 'canonical_id'
                    ) THEN
                        ALTER TABLE extracted_entity
                            ADD COLUMN canonical_id UUID
                            REFERENCES canonical_entity(canonical_id);
                    END IF;
                END
                $$;
            """))
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.warning("[migrations] canonical_id FK skipped: %s", exc)

    # ── 4. Column renames ─────────────────────────────────────────────────────
    # 'metadata' is reserved by SQLAlchemy's Declarative API.  Rename it on any
    # existing table that was created before this was caught.
    with engine.connect() as conn:
        try:
            conn.execute(text("""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE  table_name  = 'knowledge_graph_edge'
                        AND    column_name = 'metadata'
                    ) AND NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE  table_name  = 'knowledge_graph_edge'
                        AND    column_name = 'edge_metadata'
                    ) THEN
                        ALTER TABLE knowledge_graph_edge
                            RENAME COLUMN metadata TO edge_metadata;
                    END IF;
                END
                $$;
            """))
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.warning("[migrations] rename metadata→edge_metadata skipped: %s", exc)

    # ── 5. Index migrations ───────────────────────────────────────────────────
    # Replace the old HASH index on extracted_entity.entity_value with BTREE
    # (BTREE supports LIKE, range queries, and multi-column composites).
    index_stmts = [
        "DROP INDEX IF EXISTS idx_extracted_entity_value_hash",
        "CREATE INDEX IF NOT EXISTS idx_extracted_entity_value_btree  ON extracted_entity (entity_value)",
        "CREATE INDEX IF NOT EXISTS idx_extracted_entity_type_value   ON extracted_entity (entity_type, entity_value)",
        "CREATE INDEX IF NOT EXISTS idx_canonical_entity_type_value   ON canonical_entity  (entity_type, canonical_value)",
    ]
    with engine.connect() as conn:
        for stmt in index_stmts:
            try:
                conn.execute(text(stmt))
            except Exception as exc:
                logger.warning("[migrations] index stmt skipped (%s…): %s", stmt[:60], exc)
        conn.commit()

    logger.info("[migrations] Startup migrations complete.")
