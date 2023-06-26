from typing import Any, Dict, List, Mapping, Optional
from pydantic import Extra, root_validator
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun, CallbackManagerForChainRun

VALID_PROTOCOLS = ("gRPC")

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
    """Verbose"""
    verbose: bool = False
    """Response Object"""
    response: Any = None
    """Client Object"""
    client: Any

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            import grpc
            import inference_pb2_grpc

            GPRC_URL = values["host"]+":"+values["port"]
            channel = grpc.insecure_channel(GPRC_URL)
            values["client"] = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
        except ImportError:
            raise ImportError(
                "Could not import grpc proto buffers. "
                "Please install it with `pip install -U grpcio protobuf grpcio-tools`."
            )
        return values
    
    def _infer_stream(self, stub, model_input):
        import inference_pb2

        input_data = {"data": bytes(model_input, 'utf-8')}

        response = stub.StreamPredictions(
            inference_pb2.PredictionsRequest(model_name=self.model_name, input=input_data)
        )
        return response

    def _call(self, prompt, run_manager: Optional[CallbackManagerForLLMRun] = None, stop=None, **kwargs: Any) -> str:
        self.response = self._infer_stream(self.client, prompt)
        return ""

    async def _acall(self, prompt, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, stop=None, **kwargs: Any) -> str:
        if self.streaming:
            combined_text_output = ""
            self.response = self._infer_stream(self.client, prompt)
            for resp in self.response:
                prediction = resp.prediction.decode("utf-8")
                if run_manager:
                    await run_manager.on_llm_new_token(token=prediction, verbose=self.verbose)
                combined_text_output += prediction
            return combined_text_output

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"host": self.host, 
               "port": self.port, 
               "model_name": self.model_name
               },
            **{"model_kwargs": _model_kwargs},
        }
    @property
    def _llm_type(self) -> str:
        return "torchserve_endpoint"
