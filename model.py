# type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

from instill.helpers import (
    construct_text_generation_chat_infer_response,
    construct_text_generation_chat_metadata_response,
)
from instill.helpers.const import TextGenerationChatInput
from instill.helpers.ray_config import InstillDeployable, instill_deployment
from instill.helpers.ray_io import StandardTaskIO, serialize_byte_tensor


@instill_deployment
class Gemma:
    """Custom model implementation"""

    def __init__(self):
        """Load model into memory"""
        self.model = AutoModelForCausalLM.from_pretrained(
            "gemma-2-9b-it",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("gemma-2-9b-it")

    def ModelMetadata(self, req):
        """Define model input and output shape base on task type"""
        return construct_text_generation_chat_metadata_response(req=req)

    async def __call__(self, request):
        """Run inference logic"""

        # deserialize input base on task type
        task_text_generation_chat_input: TextGenerationChatInput = (
            StandardTaskIO.parse_task_text_generation_chat_input(request=request)
        )

        # preprocess
        input_ids = self.tokenizer(
            task_text_generation_chat_input.prompt, return_tensors="pt"
        ).to("cuda")

        # inference
        g_texts = self.model.generate(
            **input_ids,
            max_new_tokens=task_text_generation_chat_input.max_new_tokens,
        )
        outputs = serialize_byte_tensor(np.asarray(g_texts.cpu()))
        # sequences = self.pipeline(conv, **generation_args)

        # convert the model output into response output using StandardTaskIO
        # task_text_generation_chat_output = (
        #     StandardTaskIO.parse_task_text_generation_chat_output(sequences=sequences)
        # )

        # return response
        return construct_text_generation_chat_infer_response(
            req=request,
            # specify the output dimension
            shape=[1, len(g_texts)],
            raw_outputs=[outputs],
        )


# define model deployment entrypoint
entrypoint = InstillDeployable(Gemma).get_deployment_handle()
