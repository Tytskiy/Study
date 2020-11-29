import functools


def substitutive(f):
    @functools.wraps(f)
    def decorator(*args, **kwargs):
        if (f.__code__.co_argcount < len(args)):
            raise TypeError
        if (f.__code__.co_argcount == len(args)):
            return f(*args, **kwargs)

        @functools.wraps(f)
        def wrapper(*args_wrap, **kwargs_wrap):
            for key in kwargs.keys():
                try:
                    kwargs_wrap[key]
                except KeyError:
                    kwargs_wrap[key] = kwargs[key]
            return decorator(*(args+args_wrap), **kwargs_wrap)
        return wrapper
    return decorator
