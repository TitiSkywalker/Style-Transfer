GREEN_CHECK = "\033[32m✓\033[0m"
RED_CROSS = "\033[31m✗\033[0m"

class TaskLog:
    r"""pretty task module"""
    def __init__(self, message: str):
        super().__init__()
        self.message = message  
    
    def start(self):
        print(f"[-] {self.message}", end="\r", flush=True)

    def end(self):
        print(f"\r\033[K[{GREEN_CHECK}] {self.message}")
    
    def error(self, error):
        print(f"\r\033[K[{RED_CROSS}] {self.message}")
        print(f"{error}")

    def __enter__(self):
        self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type is None:
            self.error(exc_val)
        else:
            self.end()

class LoopLog:
    r"""pretty loop module
        - count: total number of iterations
        - precise: if set true, format is "[x/x]", otherwise format is "[x%]"
    """
    def __init__(self, message: str, count: int, precise=False):
        super().__init__()
        self.message = message  
        self.total = count
        self.current = 1
        self.precise = precise
    
    def start(self):
        if self.precise:
            print(f"[0/{self.total}] {self.message}", end="\r", flush=True)
        else:
            print(f"[0%] {self.message}", end="\r", flush=True)
        return self

    def update(self):
        self.current += 1
        if self.precise:
            print(f"\r\033[K[{self.current}/{self.total}] {self.message}", end="\r", flush=True)
        else:
            print(f"\r\033[K[{int(self.current/self.total*100)}%] {self.message}", end="\r", flush=True)

    def end(self):
        print(f"\r\033[K[{GREEN_CHECK}] {self.message}")
    
    def error(self, error):
        print(f"\r\033[K[{RED_CROSS}] {self.message}")
        print(f"{error}")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type is None:
            self.error(exc_val)
        else:
            self.end()