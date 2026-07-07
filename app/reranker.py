from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.config import settings
from app.logger import logger


class Reranker:
    def __init__(self):
        self._model = None

    def _get_model_path(self) -> str:
        model_name = settings.RERANK_MODEL
        local_dir = Path(settings.RERANK_MODEL_PATH) / model_name

        if local_dir.exists() and any(local_dir.iterdir()):
            logger.info("使用本地缓存模型: %s", local_dir)
            return str(local_dir)

        logger.info("下载重排序模型 %s 到 %s", model_name, local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_name,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            logger.info("模型下载完成: %s", local_dir)
        except Exception as e:
            logger.warning("模型下载失败 (%s)，回退到在线加载: %s", e, model_name)
            return model_name

        return str(local_dir)

    def _load_model(self):
        model_path = self._get_model_path()
        logger.info("加载重排序模型: %s", model_path)
        try:
            from sentence_transformers import CrossEncoder

            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CrossEncoder(model_path, device=device)
            logger.info("重排序模型加载完成 (device=%s)", device)
        except Exception as e:
            logger.error("重排序模型加载失败: %s", e)
            raise

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[Document]:
        if not documents:
            return []

        if self._model is None:
            self._load_model()

        top_k = top_k or settings.TOP_K

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked = [doc for doc, _ in scored_docs[:top_k]]

        if len(documents) > 1:
            high = scored_docs[0][1]
            low = (
                scored_docs[min(top_k, len(scored_docs)) - 1][1]
                if top_k <= len(scored_docs)
                else scored_docs[-1][1]
            )
            logger.info(
                "重排序完成: 候选=%d → 返回=%d, 最高分=%.4f, 最低分=%.4f",
                len(documents),
                top_k,
                high,
                low,
            )

        return reranked


reranker = Reranker()