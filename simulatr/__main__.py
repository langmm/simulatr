r"""Command-line entry point for ``python -m simulatr``."""

import sys
from .cli import main


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
