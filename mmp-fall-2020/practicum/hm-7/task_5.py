import functools


def check_arguments(*types):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if(len(types) > len(args)):
                raise TypeError
            for i in range(len(types)):
                if not isinstance(args[i], types[i]):
                    raise TypeError
            return f(*args, **kwargs)
        return wrapper

    return decorator
