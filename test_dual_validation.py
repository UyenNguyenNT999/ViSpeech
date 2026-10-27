"""Small CPU checks: python -m unittest test_dual_validation -v."""
import unittest
import numpy as np
import torch

from fixed_noise_validation import choose_checkpoints, FixedNoiseValidation, metric_values
from evaluate_noise_robustness import mix_noise


class DualValidationTests(unittest.TestCase):
    def test_loss_matches_original_ab_values_and_gradients(self):
        from train_paired import consistency_js
        from train_paired_dual_validation import paired_losses
        for weight in (0.0, 0.1):
            torch.manual_seed(42)
            logits = torch.randn(8, 3, requires_grad=True)
            labels = torch.arange(4) % 3
            log = logits.log_softmax(1)
            reference = .5 * (torch.nn.functional.nll_loss(log[:4], labels)
                              + torch.nn.functional.nll_loss(log[4:], labels))
            reference = reference + weight * consistency_js(log[:4], log[4:])
            classification, js, actual = paired_losses(log[:4], log[4:], labels, weight)
            self.assertTrue(torch.equal(reference, actual))
            expected_gradient = torch.autograd.grad(reference, logits, retain_graph=True)[0]
            actual_gradient = torch.autograd.grad(actual, logits)[0]
            self.assertTrue(torch.equal(expected_gradient, actual_gradient))

    def test_selection_and_fixed_floor(self):
        self.assertEqual(choose_checkpoints({"accuracy": .918}, .68, .913, None, None), (True, True, True))
        self.assertEqual(choose_checkpoints({"accuracy": .919}, .70, .913, .920, .68), (False, True, True))
        self.assertEqual(choose_checkpoints({"accuracy": .910}, .90, .913, .920, .70), (False, False, False))
        self.assertEqual(choose_checkpoints({"accuracy": .920}, .70, .913, .920, .70), (False, False, True))
        self.assertEqual(choose_checkpoints({"accuracy": .913}, .71, .913, .920, .70), (False, True, True))

    def test_macro_f1_not_accuracy(self):
        result = metric_values([0, 0, 1, 2], [0, 0, 0, 0], 4.0)
        self.assertEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["macro_f1"], 2/9)
        self.assertEqual(result["loss"], 1.0)

    def test_fixed_replay_no_rescaling_or_rng(self):
        fixture = FixedNoiseValidation.__new__(FixedNoiseValidation)
        fixture.count = 4
        fixture.speech = [np.array([.4, -.4, .6, -.6], dtype=np.float32)]
        fixture.noise = {"noise": np.array([2., -2., 1.], dtype=np.float32)}
        noise = np.pad(fixture.noise["noise"][1:], (0, 2), mode="wrap")
        expected, gain = mix_noise(fixture.speech[0], noise, 0.)
        row = dict(audio_id="0", noise_id="noise", noise_start_sample="1", noise_samples="4",
                   snr_db="0", noise_scale=str(gain))
        np.testing.assert_array_equal(fixture.mixed(row), expected)
        np.testing.assert_array_equal(fixture.mixed(row), fixture.mixed(row))
        row["noise_scale"] = str(gain*2)
        with self.assertRaises(ValueError):
            fixture.mixed(row)

    def test_eval_does_not_backpropagate(self):
        fixture = FixedNoiseValidation.__new__(FixedNoiseValidation)
        fixture.count = 4
        fixture.samples = [{"true_label": k} for k in range(3)]
        fixture.speech = [np.ones(4,dtype=np.float32)*(k+1)*.1 for k in range(3)]
        fixture.noise = {"n": np.array([1., -1., 1., -1.], dtype=np.float32)}
        fixture.groups = {}
        for snr in [20,10,0]:
            rows=[]
            for k,speech in enumerate(fixture.speech):
                _,gain=mix_noise(speech,fixture.noise['n'],snr)
                rows.append(dict(audio_id=str(k),noise_id='n',noise_start_sample='0',noise_samples='4',
                                 snr_db=str(snr),noise_scale=str(gain)))
            fixture.groups[('n',float(snr),0)] = rows
        model=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.LogSoftmax(dim=1))
        before={k:v.clone() for k,v in model.state_dict().items()}
        clean,mean,snrs,details=fixture.evaluate(model,torch.device('cpu'),2,2)
        self.assertEqual(len(details),3)
        self.assertAlmostEqual(mean,np.mean(list(snrs.values())))
        for k,v in model.state_dict().items():
            self.assertTrue(torch.equal(v,before[k]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
