import unittest
import pytact
from pytact import sensors, models, tasks

class TestImport(unittest.TestCase):
    
    def test_main(self):
        pytact.__version__