from abc import ABC
from enum import Enum
from typing import Any, Optional, Union
from vllm.v1.request import Request


class BudgetType(Enum):
    """Enum for budget type."""
    
    TOKEN ="token"
    CL = "computational_load"


class TokenBudget(ABC):
    """ Unified management of the token budget"""

    def __init__(self, scheduler):
        self.scheduler = scheduler

        self.type = scheduler.budget_type
        self.pp_size = scheduler.parallel_config.pipeline_parallel_size

        self.max_num_scheduled_tokens = scheduler.max_num_scheduled_tokens
        self.token_budget = self.max_num_scheduled_tokens

        self.attn_estimator = scheduler.attn_estimator

        self.max_num_scheduled_flops = self.attn_estimator.calculate_current_flops(
            chunk_size = self.max_num_scheduled_tokens,
            hist_seq_len = 0
        )
        self.flops_budget = self.max_num_scheduled_flops
        # If req need split later, modify flops_coeff.
        self.flops_coeff = 1

    def update(self):
        # update budget
        if self.type == BudgetType.TOKEN.value:
            self.token_budget = self.max_num_scheduled_tokens
        elif self.type == BudgetType.CL.value:
            self.flops_budget = self.max_num_scheduled_flops
            self.token_budget = self.max_num_scheduled_tokens

    def compute_chunk_size_with_flops(self, num_new_tokens:int, request:Request, num_computed_tokens:int, block_size:int, dcpp_min_chunk:Optional[int] = None)->int:
        dcpp_equitable_tokens = num_new_tokens
        target_tokens = min(self.flops_budget, self.max_num_scheduled_flops * self.flops_coeff)

        # Compute actual chunk size base on hist_seq_len and target_flops.
        num_new_tokens = self.attn_estimator.compute_chunk_size_with_flops(
                                                hist_seq_len = num_computed_tokens,
                                                target_flops = target_tokens,
                                                block_size = block_size)

        floor = dcpp_min_chunk if dcpp_min_chunk and dcpp_min_chunk > 0 else 0
        num_new_tokens = min(num_new_tokens, dcpp_equitable_tokens)
        num_new_tokens = min(request.num_tokens - request.num_computed_tokens,
                                max(floor, num_new_tokens))
        return num_new_tokens

    def has_running(self):
        if self.type == BudgetType.TOKEN.value:
            return self.token_budget > 0
        elif self.type == BudgetType.CL.value:
            return self.flops_budget > 0 and self.token_budget > 0

    def has_waiting(self):
        if self.type == BudgetType.TOKEN.value:
            return self.token_budget > 0
        elif self.type == BudgetType.CL.value:
            return self.flops_budget > 0 and self.token_budget > 0

    def get(self, computed_prompt: Optional[bool] = False):
        return self.token_budget

    def get_max_flops(self):
        return self.max_num_scheduled_flops

    def consume(self, num_new_tokens: int, num_computed_tokens:int, computed_prompt: Optional[bool] = False):
        if self.type == BudgetType.TOKEN.value:
            self.token_budget -= num_new_tokens
        elif self.type == BudgetType.CL.value:
            num_new_flops = self.attn_estimator.calculate_current_flops(
                chunk_size = num_new_tokens,
                hist_seq_len = num_computed_tokens
            )
            self.flops_budget -= num_new_flops
            self.token_budget -= num_new_tokens

    def rollback(self, num_new_tokens: int, num_computed_tokens:int, computed_prompt: Optional[bool] = False):
        if self.type == BudgetType.TOKEN.value:
            self.token_budget += num_new_tokens
        elif self.type == BudgetType.CL.value:
            num_new_flops = self.attn_estimator.calculate_current_flops(
                chunk_size = num_new_tokens,
                hist_seq_len = num_computed_tokens
            )
            self.flops_budget += num_new_flops
            self.token_budget += num_new_tokens

    def verify(self):
        if self.type == BudgetType.TOKEN.value:
            assert self.token_budget >= 0
        elif self.type == BudgetType.CL.value:
            assert self.token_budget >= 0
