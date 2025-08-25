from abc import ABC
from enum import Enum


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


    def update(self):
        # fixed token budget
        if self.type == BudgetType.TOKEN:
            self.token_budget = self.max_num_scheduled_tokens
        elif self.type == BudgetType.CL:
            pass

    def has_running(self):
        return self.token_budget > 0

    def has_waiting(self):
        return self.token_budget > 0

    def get(self, computed_prompt: bool):
        return self.token_budget

    def consume(self, num_new_tokens: int, computed_prompt: bool):
        if self.type == BudgetType.TOKEN:
            self.token_budget -= num_new_tokens
        elif self.type == BudgetType.CL:
            pass

    def verify(self):
        if self.type == BudgetType.TOKEN:
            assert self.token_budget >= 0
        elif self.type == BudgetType.CL:
            pass
