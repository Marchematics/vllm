# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("second_structured", [True, False])
def test_draft_tokens_handler_preserves_batch_order(second_structured: bool) -> None:
    device = torch.device("cuda")
    handler = DraftTokensHandler(device)

    batch_a = SimpleNamespace(
        req_ids=["req-a"], has_structured_output_reqs=True
    )
    batch_b = SimpleNamespace(
        req_ids=["req-b"], has_structured_output_reqs=second_structured
    )
    drafts_a = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
    drafts_b = torch.tensor([[3, 4]], dtype=torch.int32, device=device)

    handler.set_draft_tokens(batch_a, drafts_a)
    handler.set_draft_tokens(batch_b, drafts_b)

    first = handler.get_draft_tokens()
    second = handler.get_draft_tokens()

    assert first is not None
    assert first.req_ids == ["req-a"]
    assert first.draft_token_ids == [[1, 2]]

    assert second is not None
    assert second.req_ids == ["req-b"]
    expected_second = [[3, 4]] if second_structured else [[-1, -1]]
    assert second.draft_token_ids == expected_second
