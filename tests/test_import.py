"""Simple import tests for the numpy2dng package."""


def test_import_core():
    """Test that core module can be imported."""
    from numpy2dng import core

    assert hasattr(core, "DNGBASE")


def test_import_dng():
    """Test that dng module can be imported."""
    from numpy2dng import dng

    assert hasattr(dng, "DNG")
    assert hasattr(dng, "Tag")


def test_import_defs():
    """Test that defs module can be imported."""
    from numpy2dng import defs

    assert hasattr(defs, "Compression")
    assert hasattr(defs, "DNGVersion")


def test_import_packing():
    """Test that packing module can be imported."""
    from numpy2dng import packing

    assert hasattr(packing, "pack14")


def test_import_all():
    """Test importing all main modules."""
    from numpy2dng.core import DNGBASE
    from numpy2dng.dng import DNG, Tag, DNGTags
    from numpy2dng.defs import Compression, DNGVersion, SampleFormat
    from numpy2dng.packing import pack14

    assert DNGBASE is not None
    assert DNG is not None
    assert Tag is not None
    assert DNGTags is not None
    assert Compression is not None
    assert DNGVersion is not None
    assert SampleFormat is not None
    assert callable(pack14)
