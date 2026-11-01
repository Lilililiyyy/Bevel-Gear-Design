"""公开重绘入口的默认数据位置、共享源和等比例细节参数验证。"""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('replot_contact_composite_example',PROJECT/'examples/replot_contact_composite.py')
example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)


class ContactCompositeReplotEntryTests(unittest.TestCase):
    def test_default_relieved_folder_shared_sources_and_explicit_zoom(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'examples/results/relieved_contact';data.mkdir(parents=True)
            for method in ('full_gear_fe','local_halfspace'):
                for kind in ('instantaneous','sampled_cycle_maximum'):
                    (data/(method+'_'+kind+'_triangle_pressure.json.gz')).write_bytes(b'fixture; reader is mocked')
            output=root/'out'
            with patch.object(example,'PROJECT',root),patch.object(example,'load_triangle_pressure_map',return_value={'mock_only':True}) as reader,\
                    patch.object(example,'plot_contact_composite') as plotter,patch.object(sys,'argv',['replot','--output',str(output),'--instantaneous-zoom']),\
                    contextlib.redirect_stdout(io.StringIO()):example.main()
            self.assertEqual(reader.call_count,4)
            self.assertTrue(all(call.args[0].parent==data and call.kwargs=={'recompute':False} for call in reader.call_args_list))
            self.assertEqual([call.kwargs['lang'] for call in plotter.call_args_list],['zh','en'])
            self.assertTrue(all(call.args[1]==output and call.kwargs['instantaneous_zoom'] is True for call in plotter.call_args_list))
            self.assertTrue(all(call.kwargs['source_compression']=='gzip' for call in plotter.call_args_list))
            self.assertTrue(all(call.kwargs['view']=='engineering' and call.kwargs['recovery_source_file']==data/'native_pressure_recovery_sources.json.gz' for call in plotter.call_args_list))

    def test_zoom_remains_optional_for_single_language_and_explicit_data(self):
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory)/'sources';data.mkdir()
            for method in ('full_gear_fe','local_halfspace'):
                for kind in ('instantaneous','sampled_cycle_maximum'):
                    (data/(method+'_'+kind+'_triangle_pressure.json')).write_text('{}',encoding='utf8')
            with patch.object(example,'load_triangle_pressure_map',return_value={'mock_only':True}),\
                    patch.object(example,'plot_contact_composite') as plotter,\
                    patch.object(sys,'argv',['replot','--data',str(data),'--output',str(data/'out'),'--lang','en']),\
                    contextlib.redirect_stdout(io.StringIO()):example.main()
            self.assertEqual(plotter.call_count,1);self.assertEqual(plotter.call_args.kwargs['lang'],'en')
            self.assertFalse(plotter.call_args.kwargs['instantaneous_zoom'])


if __name__=='__main__':unittest.main()
