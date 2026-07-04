import unittest

from make_models import make_models, DEFAULT_LAYERS


class TestMakeModels(unittest.TestCase):
    def test_count_and_names(self):
        models = make_models(["a", "b", "c"])
        self.assertEqual(len(models), 3)
        self.assertEqual([m.name for m in models], ["a", "b", "c"])

    def test_initial_layers(self):
        models = make_models(["a", "b"])
        for m in models:
            self.assertEqual(m.layers, [128, 64, 32])

    def test_independent_layers(self):
        models = make_models(["a", "b"])
        models[0].layers.append(16)
        self.assertEqual(models[0].layers, [128, 64, 32, 16])
        self.assertEqual(models[1].layers, [128, 64, 32])

    def test_default_unchanged(self):
        original = list(DEFAULT_LAYERS)
        models = make_models(["a"])
        models[0].layers.append(99)
        self.assertEqual(DEFAULT_LAYERS, original)


if __name__ == "__main__":
    unittest.main()
