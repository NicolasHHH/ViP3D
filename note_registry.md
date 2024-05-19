
```
CONVERTERS = Registry('converter')

@CONVERTERS.register_module()
class Converter1(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

converter_cfg = dict(type='Converter1', a=a_value, b=b_value)
converter = build_from_cfg(converter_cfg, CONVERTERS)
```