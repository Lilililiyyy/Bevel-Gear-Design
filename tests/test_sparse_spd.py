"""可选稀疏引擎控制：独立稠密解、输入保护及显式资源释放。"""
from pathlib import Path
import os
import unittest
import numpy as np
from scipy import sparse
from bevel_gear.sparse_spd import PardisoSPD


class SpdInputTests(unittest.TestCase):
    def test_invalid_input_before_loading_engine(self):
        invalid = (sparse.csr_matrix([[1.,2.],[0.,1.]]),sparse.csr_matrix((2,2)),
            sparse.csr_matrix([[np.nan]]),sparse.csr_matrix([[1.+1.j]]))
        for matrix in invalid:
            with self.subTest(matrix=matrix.shape),self.assertRaises(ValueError):
                PardisoSPD(matrix,library_path='unavailable-runtime')
        for options in (dict(threads=0),dict(threads=True),dict(maximum_factor_memory_gib=np.nan)):
            with self.subTest(options=options),self.assertRaises(ValueError):
                PardisoSPD(sparse.eye(2),library_path='unavailable-runtime',**options)


@unittest.skipUnless(os.environ.get('BEVEL_MKL_RUNTIME'),'按需设置BEVEL_MKL_RUNTIME验证本机可选稀疏运行库')
class ActualSpdTests(unittest.TestCase):
    def test_independent_solution_multiple_rhs_and_close(self):
        matrix=np.array([[7.,-2.,1.],[-2.,5.,-.5],[1.,-.5,4.]])
        rhs=np.array([[1.,0.],[2.,1.],[-1.,3.]])
        with PardisoSPD(sparse.csr_matrix(matrix),library_path=Path(os.environ['BEVEL_MKL_RUNTIME']),
                maximum_factor_memory_gib=.05,threads=1) as factor:
            answer=factor.solve(rhs)
            np.testing.assert_allclose(answer,np.linalg.solve(matrix,rhs),rtol=1e-13,atol=1e-14)
            np.testing.assert_allclose(factor.solve(rhs[:,0]),answer[:,0],rtol=1e-13,atol=1e-14)
            self.assertEqual(factor.analysis_diagnostics['matrix_type'],2)
            self.assertEqual(factor.analysis_diagnostics['index_base'],0)
            with self.assertRaises(ValueError):factor.solve([1.+1.j,0.,0.])
            with self.assertRaises(ValueError):factor.solve([1.,2.])
        factor.close()
        with self.assertRaises(RuntimeError):factor.solve(rhs)

    def test_memory_analysis_limit_rejects_before_factorization(self):
        with self.assertRaises(MemoryError):
            PardisoSPD(sparse.eye(3,format='csr'),library_path=Path(os.environ['BEVEL_MKL_RUNTIME']),
                maximum_factor_memory_gib=1e-12)


if __name__=='__main__':unittest.main()
