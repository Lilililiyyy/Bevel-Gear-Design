"""法向修缘的材料去除规律、接边和参数保护；不以图形平滑证明压力收敛。"""
import json
import math
from pathlib import Path
import unittest
from bevel_gear.flank_relief import tip_relief_value
from bevel_gear.generated_surface import GeneratedGearInput


class FlankReliefLawTests(unittest.TestCase):
    def test_zero_relief_and_before_start_preserve_original_surface(self):
        for v in (0.,.2,.65,1.):
            self.assertEqual(tip_relief_value(v,0.,.65),0.)
        for v in (0.,.2,.65):
            self.assertEqual(tip_relief_value(v,30e-6,.65),0.)

    def test_tip_amount_monotonicity_and_c2_join(self):
        amount=30e-6; start=.65
        values=[tip_relief_value(i/100,amount,start) for i in range(101)]
        self.assertAlmostEqual(values[-1],amount,places=16)
        self.assertTrue(all(b>=a for a,b in zip(values,values[1:])))
        # 五次律在接边的变化为三阶；网格减半应使起始位移约缩小八倍。
        h=1e-4
        self.assertAlmostEqual(tip_relief_value(start+h,amount,start)/tip_relief_value(start+h/2,amount,start),8.,delta=.005)
        self.assertAlmostEqual((amount-tip_relief_value(1-h,amount,start))/(amount-tip_relief_value(1-h/2,amount,start)),8.,delta=.005)

    def test_invalid_geometry_parameters_are_rejected(self):
        for args in ((float('nan'),1.,.5),(.5,-1.,.5),(.5,1.,1.),(True,1.,.5),(.5,1.,-.1)):
            with self.subTest(args=args),self.assertRaises(ValueError):
                tip_relief_value(*args)
        root=Path(__file__).resolve().parents[1]
        data=json.loads((root/'examples/spiral_17_47_pinion_manufacturing.json').read_text(encoding='utf-8'))
        for key,value in (('tip_relief_m',-1.),('tip_relief_m',True),('tip_relief_start',1.)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                GeneratedGearInput(**dict(data,**{key:value}))


if __name__ == '__main__': unittest.main()
