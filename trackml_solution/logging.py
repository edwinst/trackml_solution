"""Simple helpers for printing log messages and timing info.

---

Copyright 2018 Edwin Steiner

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from time import process_time

class Logger:
    """Class for keeping track of logging state and printing messages."""
    class LogIndenter:
        """Context manager for indented log sections."""
        def __init__(self, logger):
            self._logger = logger
        def __enter__(self):
            self._logger._indent += 1
        def __exit__(self, *args):
            self._logger._indent -= 1

    class LogTimer(LogIndenter):
        """Context manager for timed and indented log sections."""
        def __init__(self, logger, caption):
            super(Logger.LogTimer, self).__init__(logger)
            self._caption = caption
        def __enter__(self):
            self._logger.log(self._caption, ":")
            super(Logger.LogTimer, self).__enter__()
            self._start_time = process_time()
        def __exit__(self, *args):
            stop_time = process_time()
            super(Logger.LogTimer, self).__exit__()
            self._logger.log("done %s: %.3fs" % (self._caption, (stop_time - self._start_time)))

    def __init__(self, max_log_indent=None):
        """
        Args:
            max_log_indent (None or int): maximum log indent level to show.
        """
        self._indent = 0
        self._max_log_indent = max_log_indent
        self._indenter = self.LogIndenter(self)

    @property
    def indent(self):
        """Returns a context manager for indenting a log section."""
        return self._indenter

    def timed(self, caption):
        """Returns a context manager for timing and indenting a log section."""
        return self.LogTimer(self, caption)

    def log(self, *args, sep='', end='\n', flush=False):
        """Print a log message (if log settings enable it)."""
        if self._max_log_indent is None or self._indent <= self._max_log_indent:
            print("    " * self._indent, *args, sep=sep, end=end, flush=flush)

