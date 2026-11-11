"""`python -m rcwe` is the same as the `rcwe` command."""
import sys

from rcwe.main import main

if __name__ == "__main__":
    sys.exit(main())
