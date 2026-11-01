"""对称稀疏排序必须保留同一三维弹性方程的响应、反力与应变能。"""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('scipy') and importlib.util.find_spec('numpy'),'需要NumPy/SciPy')
class ElasticOrderingTests(unittest.TestCase):
    def system(self,ordering,**options):
        from bevel_gear.elasticity import TetraMesh,ElasticSystem,ElasticMaterial
        nodes=[[0,0,0],[.01,0,0],[.01,.01,0],[0,.01,0],
            [0,0,.01],[.01,0,.01],[.01,.01,.01],[0,.01,.01]]
        tetra=[[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6],[0,5,1,6]]
        return ElasticSystem(TetraMesh(nodes,tetra),ElasticMaterial(210e9,.3),[0,1,2,3],
            factorization_ordering=ordering,**options)

    def test_same_matrix_multiple_loads_reactions_and_energy(self):
        import numpy as np
        first=self.system('symmetric');second=self.system('colamd')
        np.testing.assert_array_equal(first.stiffness.toarray(),second.stiffness.toarray())
        for seed in (11,19,41):
            force=np.random.default_rng(seed).normal(size=24)
            a=first.solve(force);b=second.solve(force)
            for key in ('displacements_m','reaction_forces_n','element_stress_pa'):
                np.testing.assert_allclose(a[key],b[key],rtol=3e-13,atol=2e-12)
            self.assertAlmostEqual(a['strain_energy_j']/b['strain_energy_j'],1.,places=13)
            self.assertLess(a['maximum_free_residual_n'],1e-12)

    def test_invalid_ordering_is_rejected(self):
        with self.assertRaises(ValueError):self.system('unknown')

    def test_true_cholesky_factory_has_no_fake_lu_pivot(self):
        import numpy as np
        class Cholesky:
            def __init__(self,matrix):
                self.lower=np.linalg.cholesky(matrix.toarray())
                self.positive_definite=True
                self.analysis_diagnostics=dict(backend='numpy_test',factorization='SPD_Cholesky')
            def solve(self,rhs):
                return np.linalg.solve(self.lower.T,np.linalg.solve(self.lower,rhs))
        original=self.system('symmetric');external=self.system('symmetric',factorization_factory=Cholesky)
        np.testing.assert_array_equal(original.stiffness.toarray(),external.stiffness.toarray())
        for seed in (3,9):
            force=np.random.default_rng(seed).normal(size=24)
            a=original.solve(force);b=external.solve(force)
            np.testing.assert_allclose(a['displacements_m'],b['displacements_m'],rtol=2e-13,atol=1e-22)
            self.assertLess(b['relative_free_residual'],1e-12)
        self.assertIsNone(external.diagnostics['minimum_relative_lu_pivot'])
        self.assertEqual(external.diagnostics['sparse_factorization_diagnostics']['factorization'],'SPD_Cholesky')

    def test_invalid_factory_or_inaccurate_response_cannot_be_accepted(self):
        import numpy as np
        class Inaccurate:
            positive_definite=True
            analysis_diagnostics=dict(backend='incorrect_test')
            def __init__(self,matrix):pass
            def solve(self,rhs):return np.zeros_like(rhs)
        for factory in (False,lambda matrix:object()):
            with self.subTest(factory=factory),self.assertRaises(ValueError):
                self.system('symmetric',factorization_factory=factory)
        system=self.system('symmetric',factorization_factory=Inaccurate)
        with self.assertRaisesRegex(ArithmeticError,'残差'):
            system.solve(np.ones(24))


if __name__=='__main__':unittest.main()
