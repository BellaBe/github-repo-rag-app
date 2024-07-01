import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language


def load_files(repository_path):
    loader = GenericLoader.from_filesystem(
        repository_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(
            language=Language.PYTHON
        )
    )

    docs = loader.load()

    return docs
