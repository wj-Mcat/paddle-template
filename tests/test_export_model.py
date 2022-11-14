import tempfile
import unittest

from paddlenlp.transformers import AlbertModel, BertModel, RobertaModel

from paddle_template.export_model import PLM


class TestExportModel(unittest.TestCase):
    def test_roberta_to_static(self):
        """test roberta static export"""
        model = RobertaModel.from_pretrained("hfl/rbt3")
        plm = PLM(model)
        with tempfile.TemporaryDirectory() as tempdir:
            plm.to_static(tempdir)

    def test_albert_to_static(self):
        """test roberta static export"""
        return
        model = AlbertModel.from_pretrained("albert-chinese-small")
        plm = PLM(model)
        with tempfile.TemporaryDirectory() as tempdir:
            plm.to_static(tempdir)

    def test_bert_to_static(self):
        """test bert static export"""
        model = BertModel.from_pretrained("bert-base-uncased")
        plm = PLM(model)
        with tempfile.TemporaryDirectory() as tempdir:
            plm.to_static(tempdir)
