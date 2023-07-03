from typing import Any, Dict, List, Mapping, Optional
from pydantic import Extra, root_validator
from langchain.llms.base import LLM
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForChainRun,
)

VALID_PROTOCOLS = ("REST", "gRPC")


class TorchServeEndpoint(LLM):
    """Host Address to use."""

    host: str = "localhost"
    """Port to use."""
    port: str = "7070"
    """Model name"""
    model_name: str = ""
    """Key word arguments to pass to the model."""
    model_kwargs: Optional[dict] = None
    """Streaming"""
    streaming: bool = True
    """Protocol"""
    protocol: str = "gRPC"
    """Verbose"""
    verbose: bool = False
    """Response Object"""
    response: Any = None
    """Client Object"""
    client: Any = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        URL = values["host"] + ":" + values["port"]
        try:
            if values["protocol"] == "REST":
                import requests
            elif values["protocol"] == "gRPC":
                import grpc
                import inference_pb2_grpc
            else:
                raise ValueError(
                    f'Got invalid protocol {values["protocol"]}, '
                    f"currently only {VALID_PROTOCOLS} are supported"
                )
        except ImportError as e:
            if "requests" in str(e):
                raise ImportError(
                    "Could not import requests."
                    "Please install it with `pip install -U requests`."
                )
            else:
                raise ImportError(
                    "Could not import grpc proto buffers. "
                    "Please install it with `pip install -U grpcio protobuf grpcio-tools`."
                )

        if values["protocol"] == "gRPC":
            channel = grpc.insecure_channel(URL)
            values["client"] = inference_pb2_grpc.InferenceAPIsServiceStub(
                channel
            )

        return values

    def _grpc_infer_stream(self, stub, model_input):
        import inference_pb2

        response = stub.StreamPredictions(
            inference_pb2.PredictionsRequest(
                model_name=self.model_name, input=model_input
            )
        )
        return response

    def _rest_infer_stream(self, host, port, model_input):
        import requests

        response = requests.post(
            host + ":" + port + f"/predictions/{self.model_name}",
            data=model_input,
            stream=True,
        )
        return response

    # TODO: Implement unary response for both REST and gRPC
    def _call(
        self,
        prompt,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stop=None,
        **kwargs: Any,
    ) -> str:
        print("Model Name: ", self.model_name, "Input: ", prompt)

        input_data = {"data": bytes(prompt, "utf-8")}

        if self.streaming:
            self.response = (
                self._rest_infer_stream(
                    self.host, self.port, input_data
                ).iter_content(chunk_size=None)
                if self.protocol == "REST"
                else self._grpc_infer_stream(self.client, input_data)
            )
            return ""

    # TODO: Implement unary response for both REST and gRPC
    async def _acall(
        self,
        prompt,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stop=None,
        **kwargs: Any,
    ) -> str:
        print("Model Name: ", self.model_name, "Input: ", prompt)

        input_data = {"data": bytes(prompt, "utf-8")}

        if self.streaming:
            combined_text_output = ""
            self.response = (
                self._rest_infer_stream(
                    self.host, self.port, input_data
                ).iter_content(chunk_size=None)
                if self.protocol == "REST"
                else self._grpc_infer_stream(self.client, input_data)
            )
            for resp in self.response:
                prediction = (
                    resp.decode("utf-8")
                    if self.protocol == "REST"
                    else resp.prediction.decode("utf-8")
                )
                if run_manager:
                    await run_manager.on_llm_new_token(
                        token=prediction, verbose=self.verbose
                    )
                combined_text_output += prediction
            return combined_text_output

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{
                "host": self.host,
                "port": self.port,
                "model_name": self.model_name,
                "protocol": self.protocol,
                "streaming": self.streaming,
            },
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        return "torchserve_endpoint"
