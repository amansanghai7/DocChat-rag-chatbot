-- Migration 001: Add user_id to threads for user isolation
--
-- Run this in the Supabase SQL editor (Dashboard → SQL Editor → New Query).
-- Safe to run multiple times — all statements use IF NOT EXISTS / IF EXISTS guards.
--
-- What this does:
--   1. Adds a nullable user_id column to the threads table.
--      Nullable so existing rows keep their data (no destructive changes).
--   2. Creates an index for fast user-scoped queries.
--   3. Enables Row Level Security (RLS) on all three tables as defence-in-depth.
--   4. Adds RLS policies so authenticated users can only see their own data.
--
-- NOTE: Our FastAPI backend uses the SERVICE ROLE key, which bypasses RLS.
-- Application-level filtering (WHERE user_id = ?) is the primary security mechanism.
-- RLS is a second layer of protection — useful if the anon key is ever used accidentally.

-- ─── Step 1: Add user_id column to threads ───────────────────────────────────

ALTER TABLE threads
  ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

-- Index for efficient per-user queries
CREATE INDEX IF NOT EXISTS idx_threads_user_id ON threads(user_id);

-- ─── Step 2: Enable Row Level Security ───────────────────────────────────────

ALTER TABLE threads  ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- ─── Step 3: RLS policies for threads ────────────────────────────────────────
-- auth.uid() returns the UUID of the currently authenticated Supabase user.
-- These policies run when the ANON or AUTHENTICATED role makes a request.
-- Service-role key bypasses all policies.

DROP POLICY IF EXISTS "users_select_own_threads" ON threads;
CREATE POLICY "users_select_own_threads"
  ON threads FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "users_insert_own_threads" ON threads;
CREATE POLICY "users_insert_own_threads"
  ON threads FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "users_update_own_threads" ON threads;
CREATE POLICY "users_update_own_threads"
  ON threads FOR UPDATE
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "users_delete_own_threads" ON threads;
CREATE POLICY "users_delete_own_threads"
  ON threads FOR DELETE
  USING (auth.uid() = user_id);

-- ─── Step 4: RLS policies for messages ───────────────────────────────────────
-- Messages are owned indirectly through their parent thread.

DROP POLICY IF EXISTS "users_select_own_messages" ON messages;
CREATE POLICY "users_select_own_messages"
  ON messages FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM threads
      WHERE threads.id = messages.thread_id
        AND threads.user_id = auth.uid()
    )
  );

DROP POLICY IF EXISTS "users_insert_own_messages" ON messages;
CREATE POLICY "users_insert_own_messages"
  ON messages FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM threads
      WHERE threads.id = messages.thread_id
        AND threads.user_id = auth.uid()
    )
  );

DROP POLICY IF EXISTS "users_delete_own_messages" ON messages;
CREATE POLICY "users_delete_own_messages"
  ON messages FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM threads
      WHERE threads.id = messages.thread_id
        AND threads.user_id = auth.uid()
    )
  );

-- ─── Step 5: RLS policies for documents ──────────────────────────────────────

DROP POLICY IF EXISTS "users_select_own_documents" ON documents;
CREATE POLICY "users_select_own_documents"
  ON documents FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM threads
      WHERE threads.id = documents.thread_id
        AND threads.user_id = auth.uid()
    )
  );

DROP POLICY IF EXISTS "users_insert_own_documents" ON documents;
CREATE POLICY "users_insert_own_documents"
  ON documents FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM threads
      WHERE threads.id = documents.thread_id
        AND threads.user_id = auth.uid()
    )
  );

DROP POLICY IF EXISTS "users_delete_own_documents" ON documents;
CREATE POLICY "users_delete_own_documents"
  ON documents FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM threads
      WHERE threads.id = documents.thread_id
        AND threads.user_id = auth.uid()
    )
  );
