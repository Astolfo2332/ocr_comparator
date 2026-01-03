class LlmCostManager:
    def __init__(self, input_cost: float, output_cost: float):
        """Costo por millón de tokens."""
        self.input_cost = input_cost
        self.output_cost = output_cost

    def calculate_total_cost(self, total_input, total_output) -> float:
        input_cost = (total_input/ 1_000_000) * self.input_cost
        output_cost = (total_output/ 1_000_000) * self.output_cost
        total_cost = input_cost + output_cost
        return total_cost

class CostManager:
    def __init__(self):
        self.models_costs = {
            "gpt-5.1": LlmCostManager(1.25, 10),
            "gemini-3-flash-preview": LlmCostManager(0.5, 3),
            "gemini-3-pro-preview": LlmCostManager(2, 12),
            "qwen3:32b:": LlmCostManager(0.1, 0.3),
            "qwen2.5:32b": LlmCostManager(0.09, 0.09),
            "deepseek-r1:32b": LlmCostManager(0.012, 0.18),
            "gpt-oss:20b": LlmCostManager(0.07, 0.3),
            "llama3.1:8b": LlmCostManager(0.03, 0.03),
            "deepseekOCR": LlmCostManager(0.03, 0.03),
            "HunyuanOCR": LlmCostManager(0.04, 0.04),
            "MinerU2.5": LlmCostManager(0.02, 0.02),
            "nanonets": LlmCostManager(0.05, 0.05),
        }

    def calculate_cost(self, model_name: str, total_input: int, total_output: int) -> float:
        model = self.models_costs.get(model_name)
        if not model:
            model = self.models_costs["gpt-oss:20b"]
        return model.calculate_total_cost(total_input, total_output)


cost_manager = CostManager()