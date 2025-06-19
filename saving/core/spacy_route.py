import tempfile
import os
import subprocess

import spacy
import spacy.cli.download

from logging import getLogger

logs = getLogger("huggingface_route")


def download_from_spacy(
    model_name: str,
    model_path: str,
):

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            spacy.cli.download(
                model_name,
                False,
                False,
                "--cache-dir",
                tmpdir,
            )

            ner = spacy.load(model_name)
            ner.to_disk(model_path)
        return True
    except Exception as e:
        logs.error(e, exc_info=True)
        return False
