"""错误的初始活动集也必须经全候选KKT收敛到同一有符号扭矩解。"""
import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec('numpy'),'需要NumPy')
class ContactWarmStartTests(unittest.TestCase):
    def run_case(self,initial=None,sign=-1,**options):
        import numpy as np
        from bevel_gear.fe_contact_solver import solve_fe_contact_equilibrium
        # C为一个SPD线弹性矩阵K的精确逆，保留全部非对角柔度。
        flexibility=np.eye(4)*1e-8+np.ones((4,4))*2e-9
        return solve_fe_contact_equilibrium([1e-5,0.,2e-6,3e-6],lever_arms_m=[sign*.03]*4,
            torque_nm=sign*64.,compliance_columns=lambda indices:flexibility[:,indices],
            initial_count=1,add_count=1,maximum_working_set=4,initial_indices=initial,**options)

    def test_wrong_and_correct_warm_sets_preserve_full_equilibrium(self):
        import numpy as np
        for sign in (-1,1):
            baseline=self.run_case(sign=sign)
            self.assertTrue(baseline['converged'])
            for indices in ([0],[1,2,3],[],[0,1,2,3]):
                with self.subTest(sign=sign,indices=indices):
                    warm=self.run_case(indices,sign)
                    self.assertTrue(warm['converged'])
                    np.testing.assert_allclose(warm['forces_n'],baseline['forces_n'],rtol=2e-10,atol=1e-7)
                    self.assertGreaterEqual(warm['minimum_remaining_gap_m'],-1e-10)
                    self.assertLess(abs(warm['balance_residual']),1e-8)
                    self.assertEqual(len(warm['remaining_gaps_m']),4)
            self.assertGreater(self.run_case([0])['diagnostics']['outer_history'][0]['violated_outside_working_count'],0)

    def test_invalid_or_wrong_direction_indices_are_rejected(self):
        for indices in ([True],[4],[-1],[1,1],object(),[1.5]):
            with self.subTest(indices=indices),self.assertRaises(ValueError):self.run_case(indices)
        from bevel_gear.fe_contact_solver import solve_fe_contact_equilibrium
        with self.assertRaisesRegex(ValueError,'方向'):
            solve_fe_contact_equilibrium([0.,0.],lever_arms_m=[-.03,.03],torque_nm=-1.,
                compliance_columns=lambda indices:None,initial_indices=[1])

if __name__=='__main__':unittest.main()
