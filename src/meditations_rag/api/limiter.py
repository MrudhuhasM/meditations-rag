from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize the limiter for incoming requests
# We use the remote address as the key for rate limiting
limiter = Limiter(key_func=get_remote_address)
