import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from service.common import atomic_json


class AtomicJSONTests(unittest.TestCase):
    def test_transient_windows_lock_preserves_then_replaces_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json'
            atomic_json(path, {'version':1})
            original=Path.replace
            attempts=[]
            def replace(source,target):
                attempts.append(1)
                if len(attempts)<3:
                    self.assertEqual(json.loads(path.read_text()), {'version':1})
                    raise PermissionError('temporary scanner lock')
                return original(source,target)
            with patch.object(Path,'replace',replace):
                atomic_json(path,{'version':2})
            self.assertEqual(json.loads(path.read_text()),{'version':2})
            self.assertEqual(len(attempts),3)
            self.assertEqual(list(Path(tmp).glob('*.tmp')),[])

    def test_persistent_failure_is_reported_without_losing_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json'
            atomic_json(path,{'version':1})
            with patch.object(Path,'replace',side_effect=PermissionError('locked')), patch('service.common.time.sleep'):
                with self.assertRaises(PermissionError):
                    atomic_json(path,{'version':2})
            self.assertEqual(json.loads(path.read_text()),{'version':1})
            self.assertEqual(list(Path(tmp).glob('*.tmp')),[])
