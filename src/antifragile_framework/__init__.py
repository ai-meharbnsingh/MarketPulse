
class FrameworkAPI:
    """Basic framework API for testing"""

    def __init__(self):
        self.current_provider = 'mock'

    async def get_completion(self, prompt):
        """Mock AI completion"""
        return "Mock AI response for: " + prompt[:50]

    def get_current_provider(self):
        return self.current_provider

# Make it importable
__all__ = ['FrameworkAPI']
