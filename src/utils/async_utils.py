import asyncio
import nest_asyncio

def apply_nest_asyncio():
    """
    Apply nest_asyncio to the current event loop.
    This is critical for Streamlit apps where the main thread already has a running loop.
    """
    try:
        loop = asyncio.get_running_loop()
        nest_asyncio.apply(loop)
    except RuntimeError:
        # No running loop, so nothing to patch yet
        pass

def get_or_create_event_loop():
    """
    Get the existing running loop or create a new one.
    Also ensures nest_asyncio is applied if a loop exists.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    nest_asyncio.apply(loop)
    return loop

def run_sync(coro):
    """
    Run a coroutine synchronously, handling existing event loops safely.
    Use this instead of asyncio.run() or loop.run_until_complete() directly.
    """
    loop = get_or_create_event_loop()
    if loop.is_running():
        # If loop is running, we must rely on nest_asyncio (applied in get_or_create)
        # to allow re-entrant calls.
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)
