"""
Feature Engine Pro — Logging Infrastructure

Provides a unified, environment-aware logger that renders beautifully in:
- VS Code terminals (ANSI color)
- Jupyter Lab / Notebook (HTML-formatted output)
- Google Colab (HTML-formatted output)
- Plain CI/CD terminals (clean plaintext)

The logger auto-detects the environment and adapts its output format.
"""
import logging
import sys
import io


def _is_notebook():
    """Detect if running inside a Jupyter/Colab notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell in ('ZMQInteractiveShell', 'Shell'):  # Jupyter / Colab
            return True
        return False
    except (ImportError, AttributeError, NameError):
        return False


def _supports_ansi():
    """Detect if the terminal supports ANSI escape codes."""
    if _is_notebook():
        return False
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        return True
    # VS Code integrated terminal
    import os
    if os.environ.get('TERM_PROGRAM') == 'vscode':
        return True
    return False


class _AnsiFormatter(logging.Formatter):
    """Rich ANSI-colored formatter for terminals."""
    COLORS = {
        logging.DEBUG:    '\033[90m',       # gray
        logging.INFO:     '\033[36m',       # cyan
        logging.WARNING:  '\033[33m',       # yellow
        logging.ERROR:    '\033[31m',       # red
        logging.CRITICAL: '\033[1;31m',     # bold red
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    LEVEL_ICONS = {
        logging.DEBUG:    '[DBG]',
        logging.INFO:     '[INF]',
        logging.WARNING:  '[WRN]',
        logging.ERROR:    '[ERR]',
        logging.CRITICAL: '[CRT]',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, '')
        icon = self.LEVEL_ICONS.get(record.levelno, '')
        prefix = f"{color}{self.BOLD}[FeatureEngine]{self.RESET} {icon} "
        message = record.getMessage()
        return f"{prefix}{color}{message}{self.RESET}"


class _NotebookFormatter(logging.Formatter):
    """HTML-formatted output for Jupyter/Colab using display(HTML(...))."""
    STYLES = {
        logging.DEBUG:    'color:#888;',
        logging.INFO:     'color:#00b894; font-weight:600;',
        logging.WARNING:  'color:#fdcb6e; font-weight:600;',
        logging.ERROR:    'color:#d63031; font-weight:600;',
        logging.CRITICAL: 'color:#d63031; font-weight:900; text-decoration:underline;',
    }
    LEVEL_ICONS = {
        logging.DEBUG:    '[DBG]',
        logging.INFO:     '[INF]',
        logging.WARNING:  '[WRN]',
        logging.ERROR:    '[ERR]',
        logging.CRITICAL: '[CRT]',
    }

    def format(self, record):
        style = self.STYLES.get(record.levelno, '')
        icon = self.LEVEL_ICONS.get(record.levelno, '')
        message = record.getMessage()
        return f'{icon} <span style="{style}">[FeatureEngine] {message}</span>'


class _NotebookHandler(logging.Handler):
    """Handler that uses IPython.display.HTML for notebook environments."""
    def __init__(self):
        super().__init__()
        self.setFormatter(_NotebookFormatter())

    def emit(self, record):
        try:
            from IPython.display import display, HTML
            msg = self.format(record)
            display(HTML(msg))
        except (ImportError, Exception):
            # Fallback to stderr
            sys.stderr.write(self.format(record) + '\n')


class _PlainFormatter(logging.Formatter):
    """Clean plaintext for CI/CD and piped output. ASCII-safe for Windows."""
    def format(self, record):
        msg = record.getMessage()
        # Replace common Unicode symbols with ASCII equivalents for Windows cp1252
        replacements = {
            '\u2705': '[OK]',    # ✅
            '\u274c': '[ERR]',   # ❌
            '\u26a0': '[WARN]',  # ⚠
            '\ufe0f': '',        # variation selector
            '\U0001f525': '[!!]', # 🔥
            '\U0001f50d': '[?]',  # 🔍
            '\u2191': '+',       # ↑
            '\u2193': '-',       # ↓
            '\u2192': '->',      # →
            '\u03b1': 'alpha',   # α
            '\u2265': '>=',      # ≥
            '\u2248': '~=',      # ≈
        }
        for char, replacement in replacements.items():
            msg = msg.replace(char, replacement)
        return f"[FeatureEngine] [{record.levelname}] {msg}"


def get_logger(name='feature_engine_pro', verbosity=1):
    """
    Get a configured, environment-aware logger.

    Parameters
    ----------
    name : str
        Logger name.
    verbosity : int
        0 = WARNING only (silent), 1 = INFO (summary), 2 = DEBUG (detailed).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger.setLevel(level_map.get(verbosity, logging.INFO))

    if _is_notebook():
        handler = _NotebookHandler()
    elif _supports_ansi():
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_AnsiFormatter())
    else:
        # Use a stream with error handling for Windows cp1252 terminals
        import io
        safe_stream = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace'
        ) if hasattr(sys.stdout, 'buffer') else sys.stdout
        handler = logging.StreamHandler(safe_stream)
        handler.setFormatter(_PlainFormatter())

    handler.setLevel(logger.level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
