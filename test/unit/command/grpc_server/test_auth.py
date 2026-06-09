from unittest.mock import AsyncMock, Mock

import grpc
import pytest

from command.grpc_server.auth import StaticTokenValidationInterceptor


def _details(
    metadata: tuple[tuple[str, str], ...],
    method: str = "/recommendation.v1.RecommenderService/Recommend",
) -> Mock:
    details = Mock()
    details.method = method
    details.invocation_metadata = metadata
    return details


@pytest.mark.parametrize(
    "token",
    ["Bearer secret", "secret"],
    ids=["with-bearer-prefix", "without-bearer-prefix"],
)
async def test_valid_token_passes_through(token: str) -> None:
    # Arrange
    interceptor = StaticTokenValidationInterceptor("secret")
    sentinel = object()
    continuation = AsyncMock(return_value=sentinel)

    # Act
    handler = await interceptor.intercept_service(continuation, _details((("authorization", token),)))

    # Assert
    assert handler is sentinel
    continuation.assert_awaited_once()


@pytest.mark.parametrize(
    "metadata",
    [
        (),
        (("authorization", "Bearer wrong"),),
        (("authorization", "wrong"),),
    ],
    ids=["no-token", "wrong-token-with-bearer", "wrong-token-bare"],
)
async def test_bad_token_returns_abort_handler(metadata: tuple[tuple[str, str], ...]) -> None:
    # Arrange
    interceptor = StaticTokenValidationInterceptor("secret")
    continuation = AsyncMock()

    # Act
    handler = await interceptor.intercept_service(continuation, _details(metadata))

    # Assert
    continuation.assert_not_awaited()
    assert handler is interceptor._abortion


async def test_health_check_bypasses_auth() -> None:
    # Arrange
    interceptor = StaticTokenValidationInterceptor("secret")
    sentinel = object()
    continuation = AsyncMock(return_value=sentinel)

    # Act
    handler = await interceptor.intercept_service(continuation, _details((), method="/grpc.health.v1.Health/Check"))

    # Assert
    assert handler is sentinel


def test_abort_handler_aborts_with_unauthenticated() -> None:
    # Arrange
    interceptor = StaticTokenValidationInterceptor("secret")
    context = Mock()

    # Act
    abort_behavior = interceptor._abortion.unary_unary
    assert abort_behavior is not None
    abort_behavior(None, context)

    # Assert
    context.abort.assert_called_once()
    assert context.abort.call_args.args[0] == grpc.StatusCode.UNAUTHENTICATED
