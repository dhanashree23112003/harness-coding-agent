from dotenv import load_dotenv

# Populate os.environ from .env before any test module is imported,
# so skipif conditions that check env vars evaluate correctly.
load_dotenv()
