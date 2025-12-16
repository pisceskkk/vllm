from abc import ABC
from enum import Enum
from typing import Any, Optional, Union


class BudgetType(Enum):
    """Enum for budget type."""
    
    TOKEN ="token"
    CL = "computational_load"

class TokenBudget(ABC):
    """ Unified management of the token budget
    """

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

    def update(self):
        # fixed token budget
        if self.type == BudgetType.TOKEN.value:
            self.token_budget = self.max_num_scheduled_tokens
        elif self.type == BudgetType.CL.value:
            self.flops_budget = self.max_num_scheduled_flops
            self.token_budget = self.max_num_scheduled_tokens

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

    # 这个需要看一下
    def get(self, computed_prompt: Optional[bool] = False):
        return self.token_budget

    def get_flops(self):
        return self.flops_budget

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
