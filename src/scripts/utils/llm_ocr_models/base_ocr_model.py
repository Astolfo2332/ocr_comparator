from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps
from pydantic import BaseModel
import tiktoken

def timeout(seconds=300, default=""):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fn, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except TimeoutError:
                    print("Function call timed out")
                    return default
        return wrapper
    return decorator


class BaseOcrModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_structured = None
        self.cost_model_name = ""

    def process(self, image:str) -> tuple[str, float]:
        raise NotImplementedError("Subclasses must implement this method")
    def process_structured(self,
                           image:list[str],
                           structure:BaseModel) -> tuple[BaseModel, float]:
        raise NotImplementedError("Not supported for this model")
    def start(self):
        pass
    def delete(self):
        pass

def calculate_tokens(text:str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)
