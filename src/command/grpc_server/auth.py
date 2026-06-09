import hmac
from typing import Any, Callable

import grpc
import grpc.aio


class StaticTokenValidationInterceptor(grpc.aio.ServerInterceptor):
    HEADER_KEY = "authorization"
    HEALTH_CHECK_METHOD_PREFIX = "/grpc.health.v1.Health/"

    def __init__(self, token: str) -> None:
        def abort(ignored_request: Any, context: grpc.ServicerContext) -> None:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token")

        self._abortion = grpc.unary_unary_rpc_method_handler(abort)
        self.token = token

    async def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        if handler_call_details.method.startswith(self.HEALTH_CHECK_METHOD_PREFIX):
            return await continuation(handler_call_details)

        for header, value in handler_call_details.invocation_metadata:
            if header == self.HEADER_KEY:
                token = value.decode("utf-8") if isinstance(value, bytes) else value
                if token.startswith("Bearer "):
                    token = token[7:]
                if hmac.compare_digest(self.token, token):
                    return await continuation(handler_call_details)
        return self._abortion
