"""Coordinate-system checks for the source-geometry review, without image dependencies."""
import unittest
from pathlib import Path

from datamatrix_review import compact_source_case


def event(identifier, parent, kind, numbers=None, image=None, lines=None, points=None):
    result = dict(id=identifier, parent=parent, kind=kind, numbers=numbers or {},
                  strings={}, lines=lines or [], points=points or [])
    if image:
        result.update(image=image, width=600, height=500)
    return result


class SourceGeometryReview(unittest.TestCase):
    def test_search_scales_and_source_nodes(self):
        events = [
            event(1, 0, 'source_search', {'scale': .5, 'side': 1000}, 'small.png'),
            event(2, 1, 'proposal', lines=[[[30, 60], [60, 90]]]),
            event(3, 1, 'candidate', {'crop_x': 100, 'crop_y': 200}, 'crop.png'),
            event(4, 3, 'boundaries', lines=[[[10, 20], [40, 50]]]),
            event(5, 3, 'grid', {'w': 22, 'h': 22, 'iteration': 2, 'field': 6}),
            event(6, 5, 'aligned_nodes', {'max_shift': .15}, points=[[11, 21]]),
            event(7, 5, 'rectified', {'valid': 0}, 'rectified.png', points=[[11, 21]]),
            event(8, 0, 'source_search', {'scale': .75, 'side': 1500}, 'large.png'),
            event(9, 8, 'proposal', lines=[[[30, 60], [60, 90]]]),
            event(10, 0, 'source_recovery', {'side': 1500}),
            event(11, 10, 'recovery_candidate', {'crop_x': 300, 'crop_y': 400}, 'recovery.png'),
            event(12, 11, 'refinement', {'w': 22, 'h': 22, 'iteration': 3, 'field': 8}),
            event(13, 12, 'aligned_nodes', {'max_shift': .2}, points=[[21, 31]]),
            event(14, 0, 'result', {'width': 2000, 'height': 1000}),
        ]
        base = Path(__file__).resolve().parent
        case = compact_source_case('dm0001.jpg', events, base, base, {}, {'decoded': False})
        first, second = case['passes']
        self.assertEqual([first['id'], second['id']], [1, 8])
        self.assertEqual(first['boxes'][0]['points'][0], [60, 120])
        self.assertEqual(second['boxes'][0]['points'][0], [40, 80])
        self.assertEqual(first['boxes'][1]['points'][0], [110, 220])
        self.assertEqual(first['selected'], [3])
        self.assertEqual(second['selected'], [])
        views = {v['id']: v for v in case['views']}
        self.assertEqual(views[6]['image'], 'crop.png')
        self.assertEqual(views[6]['points'], [[11, 21]])
        self.assertEqual(views[6]['numbers']['iteration'], 2)
        self.assertEqual(views[6]['numbers']['field'], 6)
        self.assertEqual(views[6]['pass'], 1)
        self.assertEqual(views[7]['image'], 'rectified.png')
        self.assertEqual(views[7]['points'], [])
        self.assertEqual(views[13]['pass'], 8)
        self.assertEqual(views[13]['candidate'], 11)
        self.assertEqual(views[13]['image'], 'recovery.png')
        self.assertEqual(views[13]['points'], [[21, 31]])
        self.assertEqual(views[13]['numbers']['iteration'], 3)
        self.assertEqual(views[13]['numbers']['field'], 8)


if __name__ == '__main__':
    unittest.main()
