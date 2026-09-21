"""PEP 440 version comparisons with infix operators.

torchada gates a number of patches on the installed ``torch_musa`` version.
Spelling those gates as dedicated predicates (``_version_lt``/``_is_pre_*``)
does not scale: every new gate needs a new helper, the bound is buried in a
function body, and the call site cannot say what it is comparing against. This
module provides a small proxy instead, so a gate reads like the relationship it
encodes::

    from ._version import version_of

    if version_of(getattr(torch.musa, "__version__", None)) < "2.11.0.post2":
        ...  # workaround for everything older than the fix

``version_of`` accepts a version string, ``None``, an object exposing
``__version__`` (a module, for example), or another proxy, and the proxy
supports ``<``, ``<=``, ``>``, ``>=``, ``==`` and ``!=`` against any of those,
in either operand order.

Two policies are implemented here so that they do not have to be repeated at
every call site:

* **The local version segment is ignored.** ``2.11.0.post1+musa5.2.0`` and
  ``2.11.0.post1`` compare equal, because the ``+musa*`` suffix identifies the
  MUSA stack build rather than the level of the fix being gated. Comparisons
  therefore use the *public* version.
* **An unknown or unparsable version compares as the lowest possible version**
  (equivalent to ``0``). A gate written as an upper bound - the shape used by
  torchada's existing gates - is then ``True`` for an unknown version, which is
  exactly the "keep the workaround when the version cannot be trusted"
  behaviour those gates document. A gate that must *skip* work for an old
  release has to say so explicitly, hence ``is_known``.

Gates spell their bound inline (``version_of(module) >= "2.11.0.post2"``), and this
proxy stays the comparator for that shape because it is parse-failure safe: a
malformed ``__version__`` must not raise (``packaging.version.parse`` would abort
at import time and take the patch with it) and must rank below every bound, so the
workaround stays armed instead of silently disappearing.
"""

from __future__ import annotations

from typing import Any

__all__ = ["VersionComparison", "version_of"]

_LOWEST = "0"


def _public_version(raw: Any):
    """Import PyTorch's vendored PEP 440 parser and parse ``raw``.

    Returns ``None`` when the version is missing, malformed, or when the parser
    itself is unavailable (a torch build without the vendored copy).
    """
    if raw is None:
        return None
    public = str(raw).split("+", 1)[0].strip()
    if not public:
        return None
    try:
        from torch._vendor.packaging.version import InvalidVersion, Version
    except ImportError:  # pragma: no cover - depends on the torch build
        return None
    try:
        return Version(public)
    except InvalidVersion:
        return None


def _raw_version(source: Any) -> Any:
    """Return the version string carried by ``source``, if any."""
    if isinstance(source, VersionComparison):
        return source.raw
    if source is None or isinstance(source, str):
        return source
    return getattr(source, "__version__", None)


class VersionComparison:
    """A comparable view of a package version, driven by infix operators."""

    __slots__ = ("_raw", "_parsed")

    def __init__(self, source: Any):
        self._raw = _raw_version(source)
        self._parsed = _public_version(self._raw)

    @property
    def raw(self) -> Any:
        """The version string this proxy was built from (``None`` if unknown)."""
        return self._raw

    @property
    def is_known(self) -> bool:
        """Whether the version parsed into a comparable PEP 440 version."""
        return self._parsed is not None

    def _rank(self) -> Any:
        """The parsed version, or the lowest version for anything unknown."""
        return self._parsed if self._parsed is not None else _public_version(_LOWEST)

    def _compare(self, other: Any) -> int:
        mine = self._rank()
        theirs = _public_version(_raw_version(other))
        if theirs is None:
            theirs = _public_version(_LOWEST)
        if mine == theirs:
            return 0
        return -1 if mine < theirs else 1

    def __lt__(self, other: Any) -> bool:
        return self._compare(other) < 0

    def __le__(self, other: Any) -> bool:
        return self._compare(other) <= 0

    def __gt__(self, other: Any) -> bool:
        return self._compare(other) > 0

    def __ge__(self, other: Any) -> bool:
        return self._compare(other) >= 0

    def __eq__(self, other: Any) -> bool:
        return self._compare(other) == 0

    def __ne__(self, other: Any) -> bool:
        return self._compare(other) != 0

    def __hash__(self) -> int:
        return hash(self._rank())

    def __repr__(self) -> str:
        shown = repr(self._raw) if self.is_known else f"unknown ({self._raw!r})"
        return f"VersionComparison({shown})"


def version_of(source: Any) -> VersionComparison:
    """Return a comparable version for ``source``.

    ``source`` may be a version string, ``None``, an object with a
    ``__version__`` attribute (for example ``torch.musa`` or a module), or an
    existing :class:`VersionComparison`.
    """
    if isinstance(source, VersionComparison):
        return source
    return VersionComparison(source)
