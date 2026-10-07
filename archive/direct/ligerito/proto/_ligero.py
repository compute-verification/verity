"""Import shim for the reused ``archive/direct/ligero`` modules: ``archive.direct`` is a namespace directory, so
under pytest (rootdir ``archive/direct``) the sibling package is ``ligero``, under ``python -m archive.direct...``
it is ``archive.direct.ligero``."""
try:
    from archive.direct.ligero import encode_simt, field, merkle          # noqa: F401
except ImportError:                                                        # pragma: no cover
    from ligero import encode_simt, field, merkle                          # noqa: F401

try:
    try:
        from archive.direct.ligero import commit_gpu                      # noqa: F401
    except ImportError:                                                    # pragma: no cover
        from ligero import commit_gpu                                      # noqa: F401
except Exception:  # noqa: BLE001 -- no cupy: the host / hash_gpu paths stay
    commit_gpu = None
