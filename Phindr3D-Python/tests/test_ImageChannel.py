import unittest
import sys
import os

# Add both the src directory and the src/Data directory to the path
# This allows 'from Data.ImageChannel import ImageChannel' to work if PYTHONPATH includes src
# AND 'from ImageChannel import ImageChannel' to work if we want to bypass the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/Data')))

# We use a try-except block to prefer the standard package import,
# but fall back to direct import if dependencies of the package are missing.
try:
    from Data.ImageChannel import ImageChannel
except (ImportError, ModuleNotFoundError):
    from ImageChannel import ImageChannel

class TestImageChannel(unittest.TestCase):
    def test_initialization(self):
        """Test that ImageChannel initializes with an empty channelpath."""
        channel = ImageChannel()
        self.assertEqual(channel.channelpath, "")

    def test_setPath(self):
        """Test that setPath correctly updates the channelpath."""
        channel = ImageChannel()
        test_path = "/path/to/image.tif"
        channel.setPath(test_path)
        self.assertEqual(channel.channelpath, test_path)

    def test_setPath_empty_string(self):
        """Test setPath with an empty string."""
        channel = ImageChannel()
        channel.setPath("")
        self.assertEqual(channel.channelpath, "")

    def test_setPath_none(self):
        """Test setPath with None."""
        channel = ImageChannel()
        channel.setPath(None)
        self.assertEqual(channel.channelpath, None)

    def test_setPath_long_path(self):
        """Test setPath with a long path string."""
        channel = ImageChannel()
        long_path = "a" * 1000
        channel.setPath(long_path)
        self.assertEqual(channel.channelpath, long_path)

if __name__ == '__main__':
    unittest.main()
