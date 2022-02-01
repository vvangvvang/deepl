class IterRegistry(type):
    def __iter__(cls):
        return iter(cls._registry)

class Person(object):
    __metaclass__ = IterRegistry
    _registry = []
    def __init__(self, name):
        self._registry.append(self)
        self.name = name

p = Person('John')
p2 = Person('Mary')
print(p._registry[0].name)
