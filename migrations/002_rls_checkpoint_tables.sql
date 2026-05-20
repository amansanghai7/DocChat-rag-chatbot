-- Migration 002: Enable RLS on LangGraph checkpoint tables
--
-- Run this in: Supabase Dashboard → SQL Editor → New Query → Run
-- Safe to run multiple times (ENABLE ROW LEVEL SECURITY is idempotent).
--
-- BACKGROUND
-- ----------
-- These 4 tables are auto-created by LangGraph's PostgresSaver.setup().
-- LangGraph has no awareness of Supabase's security model, so it creates
-- them without RLS, leaving them fully accessible via Supabase's PostgREST
-- REST API to anyone with the anon key.
--
-- WHAT THIS MIGRATION DOES
-- ------------------------
-- Enables RLS with NO permissive policies on each table.
-- "RLS enabled + no policies" = deny all access via PostgREST/REST API.
-- This silences the Supabase security alert and closes the exposure.
--
-- WHY LANGGRAPH IS NOT AFFECTED
-- ------------------------------
-- LangGraph connects via a direct psycopg TCP connection to PostgreSQL
-- using the `postgres` superuser (DATABASE_URL / session pooler).
-- PostgreSQL superusers ALWAYS bypass RLS — it is a database-level
-- guarantee. No policy we add here can block the postgres user.
-- checkpointer.setup(), get(), put(), list() all continue working.
--
-- NO application code changes or container restarts are needed.

ALTER TABLE public.checkpoints          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.checkpoint_blobs     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.checkpoint_writes    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.checkpoint_migrations ENABLE ROW LEVEL SECURITY;
