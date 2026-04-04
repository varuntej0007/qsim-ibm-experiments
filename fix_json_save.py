"""
Fix JSON serialization — run after any script that fails on JSON save
Converts numpy types to Python native types
"""
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, bool): return bool(obj)
        return super().default(obj)

# Test it
test = {"val": np.int32(4), "arr": np.array([1.0,2.0]), "flag": np.bool_(True)}
print(json.dumps(test, cls=NumpyEncoder))
print("Encoder working")
