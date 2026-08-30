# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque

import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        self._pending: deque[
            tuple[list[str], np.ndarray | None, int, torch.cuda.Event | None]
        ] = deque()

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        req_ids = list(input_batch.req_ids)
        num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs:
            # Keep this batch in the hand-off order even though no draft token
            # validation is needed for it.
            self._pending.append((req_ids, None, num_draft_tokens, None))
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        copy_event = torch.cuda.Event(blocking=True)
        with torch.cuda.stream(self.copy_stream):
            draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            copy_event.record()
        self._pending.append(
            (req_ids, draft_tokens_np, num_draft_tokens, copy_event)
        )

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if not self._pending:
            return None

        req_ids, draft_tokens_np, num_draft_tokens, copy_event = self._pending.popleft()
        if draft_tokens_np is not None:
            assert copy_event is not None
            copy_event.synchronize()
            draft_token_ids = draft_tokens_np.tolist()
        else:
            draft_token_ids = [[-1] * num_draft_tokens for _ in req_ids]
        return DraftTokenIds(req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
