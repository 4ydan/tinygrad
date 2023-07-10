import unittest
import numpy as np
from typing import List
from tinygrad.helpers import getenv, DType, DEBUG
from tinygrad.lazy import Device
from tinygrad.tensor import Tensor, dtypes
from extra.utils import OSX

def _test_to_np(a:Tensor, np_dtype, target: List[int]):
  print(a)
  na = a.numpy()
  print(na, na.dtype, a.lazydata.realized)
  assert na.dtype == np_dtype
  np.testing.assert_allclose(na, target)

def _test_op(fxn, target_dtype:DType, target: List[int]):
  c = fxn()
  if DEBUG >= 2: print(c.numpy())
  assert c.dtype == target_dtype
  np.testing.assert_allclose(c.numpy(), target)

def _test_cast(a:Tensor, target_dtype:DType, target: List[int]): _test_op(lambda: a.cast(target_dtype), target_dtype, target)
def _test_add(a:Tensor, b:Tensor, target_dtype:DType, target: List[int]): _test_op(lambda: a+b, target_dtype, target)
def _test_mul(a:Tensor, b:Tensor, target_dtype:DType, target: List[int]): _test_op(lambda: a*b, target_dtype, target)
def _test_matmul(a:Tensor, b:Tensor, target_dtype:DType, target: List[int]): _test_op(lambda: a@b, target_dtype, target)
def _test_add_upcast(a:Tensor, b:Tensor, target_dtype:DType, target: List[int]): _test_op(lambda: a+b, target_dtype, target)
def _test_mul_upcast(a:Tensor, b:Tensor, target_dtype:DType, target: List[int]): _test_op(lambda: a*b, target_dtype, target)
def _test_matmul_upcast(a:Tensor, b:Tensor, target_dtype:DType, target: List[int]): _test_op(lambda: a@b, target_dtype, target)

# for GPU, cl_khr_fp16 isn't supported (except now we don't need it!)
# for LLVM, it segfaults because it can't link to the casting function
@unittest.skipIf(getenv("CI", "") != "" and Device.DEFAULT in ["LLVM"], "float16 broken in some CI backends")
class TestHalfDtype(unittest.TestCase):
  def test_half_to_uint8(self):
    _test_cast(Tensor([1,2,3,4], dtype=dtypes.float16), dtypes.uint8, [1,2,3,4])
 

if __name__ == '__main__':
  unittest.main()
