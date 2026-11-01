"""含固定自由度反力的独立稠密控制，核对原W/u/F归档及失配拒绝。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from scipy import sparse
from bevel_gear.native_fe_archive import save_native_fe_response,file_sha256


def fixture():
    forces=np.array([2.,3.]);systems={};matrices={};fields={};energy=0.
    for j,member in enumerate(('pinion','wheel')):
        k=(j+2)*np.diag(np.arange(1.,7.))+np.ones((6,6))*.1
        w=np.array([[.2,.3],[1.,0.],[0.,1.],[.5,.2],[.1,.4],[.3,.1]])*(j+1)
        load=w@forces;u=np.r_[0.,np.linalg.solve(k[1:,1:],load[1:])]
        systems[member]=SimpleNamespace(stiffness=sparse.csr_matrix(k),free_dofs=np.arange(1,6))
        matrices[member]=sparse.csc_matrix(w)
        fields[member]=dict(displacements_m=u.reshape(-1,3),nodal_forces_n=load.reshape(-1,3))
        energy+=.5*float(u@k@u)
    prepared=SimpleNamespace(systems=systems)
    system=dict(records=[dict(tooth_offset=0,row=1,column=2),dict(tooth_offset=-1,row=3,column=4)],matrices=matrices)
    result=dict(equilibrium=dict(forces_n=forces,strain_energy_j=energy),body_fields=fields)
    return prepared,system,result


class ResponseArchiveTests(unittest.TestCase):
    def test_original_arrays_and_fixed_reaction(self):
        prepared,system,result=fixture()
        with TemporaryDirectory() as temporary:
            folder=Path(temporary)
            receipt=save_native_fe_response(prepared,system,result,folder,phase_index=7,input_fingerprint='a'*64)
            self.assertTrue(receipt['passed']);self.assertLess(receipt['condensed_energy_relative_error'],1e-14)
            for member,row in receipt['bodies'].items():
                path=folder/row['response_file'];self.assertEqual(file_sha256(path),row['response_sha256'])
                with np.load(path,allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['displacements_m'],result['body_fields'][member]['displacements_m'])
                    np.testing.assert_array_equal(data['nodal_forces_n'],result['body_fields'][member]['nodal_forces_n'])
                    np.testing.assert_array_equal(data['contact_cell_keys'],[[0,1,2],[-1,3,4]])
                    self.assertNotEqual(float(data['ku_minus_f_n'].ravel()[0]),0.)
                    self.assertLess(np.linalg.norm(data['ku_minus_f_n'].ravel()[1:]),1e-14)
                loaded=sparse.load_npz(folder/row['load_matrix_file'])
                np.testing.assert_array_equal(loaded.toarray(),system['matrices'][member].toarray())
            with self.assertRaises(FileExistsError):
                save_native_fe_response(prepared,system,result,folder,phase_index=7,input_fingerprint='a'*64)

    def test_wrong_projection_or_equilibrium_does_not_create_archive(self):
        for mode in ('W','u','energy'):
            prepared,system,result=fixture()
            if mode=='W':system['matrices']['pinion']=system['matrices']['pinion']*1.01
            elif mode=='u':result['body_fields']['pinion']['displacements_m'][0,1]*=1.01
            else:result['equilibrium']['strain_energy_j']*=1.01
            with self.subTest(mode=mode),TemporaryDirectory() as temporary:
                with self.assertRaises(ArithmeticError):
                    save_native_fe_response(prepared,system,result,temporary,phase_index=0,input_fingerprint='a'*64)
                self.assertFalse((Path(temporary)/'responses').exists())

    def test_bad_input_identity_rejected(self):
        prepared,system,result=fixture()
        with TemporaryDirectory() as temporary,self.assertRaises(ValueError):
            save_native_fe_response(prepared,system,result,temporary,phase_index=0,input_fingerprint='not-a-sha')

    def test_nan_operator_cannot_pass_residual_comparisons(self):
        prepared,system,result=fixture();system['matrices']['pinion'].data[0]=np.nan
        with TemporaryDirectory() as temporary,self.assertRaises(ValueError):
            save_native_fe_response(prepared,system,result,temporary,phase_index=0,input_fingerprint='a'*64)


if __name__=='__main__':unittest.main()
