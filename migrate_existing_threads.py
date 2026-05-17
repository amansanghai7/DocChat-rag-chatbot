"""
One-time migration helper.

Run this script after switching to Supabase to back-fill title rows for any
LangGraph checkpoint threads that don't yet have a matching row in the
Supabase `threads` table.

Usage:
    python migrate_existing_threads.py
"""

from langchain_core.messages import HumanMessage

import db_service
from langgraph_rag_backend import chatbot, checkpointer, generate_chat_title


def get_first_user_message(thread_id: str) -> str | None:
    """Return the first HumanMessage content from a thread's LangGraph state."""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        for msg in state.values.get("messages", []):
            if isinstance(msg, HumanMessage):
                return msg.content
        return None
    except Exception as e:
        print(f"  ⚠️  Could not load state for thread {thread_id[:8]}…: {e}")
        return None


def migrate_threads() -> None:
    print("🔄 Starting thread migration to Supabase…\n")

    # Collect all thread IDs known to LangGraph
    all_checkpoint_ids: set[str] = set()
    try:
        for checkpoint in checkpointer.list(None):
            tid = (checkpoint.config.get("configurable") or {}).get("thread_id")
            if tid:
                all_checkpoint_ids.add(str(tid))
    except Exception as e:
        print(f"❌ Could not read LangGraph checkpoints: {e}")
        return

    print(f"📊 Found {len(all_checkpoint_ids)} thread(s) in LangGraph checkpoints")

    # Collect thread IDs already registered in Supabase
    existing_threads = db_service.get_all_threads()
    existing_ids = {t["thread_id"] for t in existing_threads}
    print(f"✅ {len(existing_ids)} thread(s) already in Supabase\n")

    to_migrate = all_checkpoint_ids - existing_ids
    if not to_migrate:
        print("✨ Nothing to migrate — all threads are already in Supabase.")
        return

    print(f"🔧 Migrating {len(to_migrate)} thread(s)…\n")

    migrated = failed = 0
    for thread_id in to_migrate:
        print(f"  Processing {thread_id[:8]}…")
        first_msg = get_first_user_message(thread_id)

        if first_msg:
            title = generate_chat_title(first_msg)
            print(f"    Generated title: '{title}'")
        else:
            title = "New Chat"
            print(f"    No messages found — using default title")

        result = db_service.create_thread(thread_id, title)
        if result:
            migrated += 1
            print(f"    ✅ Migrated\n")
        else:
            failed += 1
            print(f"    ❌ Failed to insert into Supabase\n")

    print("=" * 60)
    print(f"  ✅ Migrated:  {migrated}")
    print(f"  ❌ Failed:    {failed}")
    print("=" * 60)

    if migrated:
        print("\n✨ Migration complete. Restart Streamlit to see the changes.")
    if failed:
        print(f"\n⚠️  {failed} thread(s) failed. Check the output above.")


if __name__ == "__main__":
    try:
        migrate_threads()
    except KeyboardInterrupt:
        print("\n\n⚠️  Migration interrupted by user.")
    except Exception as e:
        import traceback
        print(f"\n\n❌ Migration failed: {e}")
        traceback.print_exc()
