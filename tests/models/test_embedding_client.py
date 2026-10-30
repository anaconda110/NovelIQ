from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError

from src.embedding import EmbeddingClient, EmbeddingServiceError, format_query_instruct


def _build_httpx_response(status_code: int) -> httpx.Response:
    """测试里统一用这个 helper 生成最小可用响应，避免每个用例重复拼装 request/response。"""
    request = httpx.Request("POST", "http://embedding.test/v1/embeddings")
    return httpx.Response(status_code=status_code, request=request)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("BAAI/bge-m3", "林渡 顾霜"),
        ("BGE-large-zh-v1.5", "林渡 顾霜"),
        ("Qwen/Qwen3-Embedding-0.6B", format_query_instruct("林渡 顾霜")),
        ("qwen3-embedding-8b", format_query_instruct("林渡 顾霜")),
        ("text-embedding-3-small", "林渡 顾霜"),
    ],
)
def test_format_query_uses_model_aware_rules(model: str, expected: str) -> None:
    """BGE 和未知模型保留原文，Qwen Embedding 使用 instruct 前缀，且模型名大小写不敏感。"""
    with patch("src.embedding.AsyncOpenAI"):
        client = EmbeddingClient(
            base_url="http://embedding.test/v1",
            model=model,
            api_key="test-key",
            timeout_s=1.0,
        )

    assert client.format_query("林渡 顾霜") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B"])
async def test_embed_texts_never_formats_documents(model: str) -> None:
    """文档侧无论使用 BGE 还是 Qwen，都必须向嵌入服务发送原文。"""
    with patch("src.embedding.AsyncOpenAI") as openai_mock:
        mock_client = MagicMock()
        openai_mock.return_value = mock_client
        response = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])],
            usage=None,
        )
        create_mock = AsyncMock(return_value=response)
        mock_client.embeddings.create = create_mock
        client = EmbeddingClient(
            base_url="http://embedding.test/v1",
            model=model,
            api_key="test-key",
            timeout_s=1.0,
        )

        await client.embed_texts(["林渡与顾霜并肩迎敌。"])

    assert create_mock.await_args.kwargs["input"] == ["林渡与顾霜并肩迎敌。"]


@pytest.mark.asyncio
async def test_embed_texts_retries_retryable_502_and_recovers() -> None:
    """瞬时 502 属于 provider/gateway 常见抖动；batch embedding 应有限重试，而不是首次失败就终止整条 preprocess。"""
    client = EmbeddingClient(
        base_url="http://localhost:9999/v1",
        model="test-embedding",
        api_key="test-key",
        timeout_s=1.0,
    )
    success_response = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
            SimpleNamespace(index=1, embedding=[0.3, 0.4]),
        ],
        usage=None,
    )
    create_mock = AsyncMock(
        side_effect=[
            InternalServerError("bad gateway", response=_build_httpx_response(502), body=None),
            success_response,
        ]
    )
    client._client = SimpleNamespace(embeddings=SimpleNamespace(create=create_mock))

    with patch("src.embedding.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        embeddings = await client.embed_texts(["第一段", "第二段"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert create_mock.await_count == 2
    sleep_mock.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
async def test_embed_texts_does_not_retry_bad_request() -> None:
    """4xx 参数错误不属于瞬时故障，必须继续 fail fast，避免无意义重试把真实配置问题掩盖掉。"""
    client = EmbeddingClient(
        base_url="http://localhost:9999/v1",
        model="test-embedding",
        api_key="test-key",
        timeout_s=1.0,
    )
    create_mock = AsyncMock(
        side_effect=BadRequestError("bad request", response=_build_httpx_response(400), body={"error": "invalid"})
    )
    client._client = SimpleNamespace(embeddings=SimpleNamespace(create=create_mock))

    with patch("src.embedding.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with pytest.raises(EmbeddingServiceError):
            await client.embed_texts(["第一段"])

    assert create_mock.await_count == 1
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get_embedding", "embed_texts"])
@pytest.mark.parametrize("timeout", [False, True])
async def test_embedding_transport_failure_has_dependency_and_builtin_types(method: str, timeout: bool) -> None:
    """连接级重试：连接及超时批内最多重试 3 次，耗尽仍保留原有异常语义"""
    from src.embedding import EMBEDDING_CONNECTION_MAX_RETRIES

    request = httpx.Request("POST", "http://embedding.test/v1/embeddings")
    failure = APITimeoutError(request=request) if timeout else APIConnectionError(request=request)
    client = EmbeddingClient(base_url="http://embedding.test/v1", model="test", api_key="test", timeout_s=1)
    with patch.object(client._client.embeddings, "create", new=AsyncMock(side_effect=failure)) as create:
        with patch("src.embedding.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            with pytest.raises(EmbeddingServiceError) as error:
                await getattr(client, method)(["正文"] if method == "embed_texts" else "正文")
    assert isinstance(error.value, TimeoutError if timeout else ConnectionError)
    assert error.value.__cause__ is failure
    assert create.await_count == 1 + EMBEDDING_CONNECTION_MAX_RETRIES
    assert sleep_mock.await_count == EMBEDDING_CONNECTION_MAX_RETRIES
    await client._client.close()


@pytest.mark.asyncio
async def test_exhausted_embedding_http_retries_are_dependency_failure() -> None:
    """传输层有限重试耗尽后明确失败，不转成模型可修正的参数错误"""
    client = EmbeddingClient(base_url="http://embedding.test/v1", model="test", api_key="test", timeout_s=1)
    failure = InternalServerError("service unavailable", response=_build_httpx_response(503), body=None)
    with (
        patch.object(client._client.embeddings, "create", new=AsyncMock(side_effect=failure)) as create,
        patch("src.embedding.asyncio.sleep", new=AsyncMock()),
        pytest.raises(EmbeddingServiceError),
    ):
        await client.get_embedding("正文")
    assert create.await_count == 3
    await client._client.close()
