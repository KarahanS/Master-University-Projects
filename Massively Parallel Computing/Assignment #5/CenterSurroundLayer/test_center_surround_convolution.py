import unittest
import numpy as np
import os
import torch
import center_surround_convolution as csc
from torch.utils.cpp_extension import load

# The tests will only work after you finished tasks c) - e)
center_surround = load(name="center_surround_cuda",
                       sources=["center_surround_convolution.cu"],
                       extra_cuda_cflags=["-lineinfo",
                                          "--resource-usage"],
                       verbose=True)


def pytorch_center_surround(I: torch.Tensor, w_c: torch.tensor,
                            w_s: torch.Tensor, w_b: torch.Tensor):
        C_i = w_s.size(0)
        C_o = w_s.size(1)
        weight_surround = torch.ones(C_o, C_i, 3, 3,
                                     device=w_s.get_device())
        weight_surround_center = torch.ones(C_o, C_i, 1, 1,
                                            device=w_s.get_device())
        weight_center = torch.ones(C_o, C_i, 1, 1,
                                   device=w_c.get_device())
        weight_surround = (weight_surround *
                           w_s.view(C_i, C_o, 1, 1).transpose(0, 1))
        weight_surround_center = (weight_surround_center *
                                  w_s.view(C_i, C_o, 1, 1).transpose(0, 1))
        weight_center = (weight_center *
                         w_c.view(C_i, C_o, 1, 1).transpose(0, 1))
        result = torch.nn.functional.conv2d(I, weight_surround)
        result -= torch.nn.functional.conv2d(I[:, :, 1:-1, 1:-1],
                                             weight_surround_center)
        result += torch.nn.functional.conv2d(I[:, :, 1:-1, 1:-1],
                                             weight_center)
        result += w_b[None, :, None, None]
        return result


class TestCenterSurroundConvolution(unittest.TestCase):
    def test_bias_forward(self):
        device = torch.device("cuda")
        I = torch.randn(12, 3, 123, 421).to(device)
        w_c = torch.zeros(3, 64, device=device)
        w_s = torch.zeros(3, 64, device=device)
        w_b = torch.ones(64, device=device)
        O = center_surround.forward(I, w_c, w_s, w_b)[0]
        mean = torch.mean(O)
        self.assertEqual(mean.item(), 1.0,
                         msg="Bias in forward pass not added correctly!")

    def test_forward_copy(self):
        # Test if input can be copied with center weights
        device = torch.device("cuda")
        batch_size = 4
        for size in [2, 32, 64, 128, 500]:
            I = torch.randn(batch_size, 3, size, size).to(device)
            w_c = torch.from_numpy(np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)).to(device)
            w_s = torch.zeros_like(w_c)
            w_b = torch.zeros(3, device=device)
            O = center_surround.forward(I, w_c, w_s, w_b)[0]
            difference = torch.sum(I[:, :, 1: size-1, 1: size-1] - O)
            self.assertAlmostEqual(difference.item(), 0.0,
                                   msg="Center weigth influence is wrong in "
                                   "forward pass.")

    def test_forward_invert(self):
        # test inversion of input with center weights
        device = torch.device("cuda")
        batch_size = 4
        for size in [2, 32, 64, 128, 500]:
            I = torch.randn(batch_size, 3, size, size).to(device)
            w_c = torch.from_numpy(np.array(
                [[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)).to(device)
            w_s = torch.zeros_like(w_c)
            w_b = torch.zeros(3, device=device)
            O = center_surround.forward(I, w_c, w_s, w_b)[0]
            I = I[:, :, 1: size-1, 1: size-1]
            difference = torch.sum(I + O)
            self.assertAlmostEqual(difference.item(), 0.0,
                                   msg="Center weight influence is wrong in "
                                   "forward pass.")

    def test_forward_color_change(self):
        # test swap of color chanel with center weights
        device = torch.device("cuda")
        batch_size = 4
        for size in [2, 32, 64, 128, 500]:
            I = torch.randn(batch_size, 3, size, size).to(device)
            w_c = torch.from_numpy(np.array(
                [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                dtype=np.float32)).to(device)
            w_s = torch.zeros_like(w_c)
            w_b = torch.zeros(3, device=device)
            O = center_surround.forward(I, w_c, w_s, w_b)[0]
            I = I[:, :, 1: size-1, 1: size-1]
            expected_result = torch.unbind(I, 1)
            expected_result = torch.stack(expected_result[::-1], 1)
            difference = torch.sum(expected_result - O)
            self.assertAlmostEqual(difference.item(), 0.0,
                                   msg="Centwer weight influence is wrong in "
                                   "forward pass.")

    def test_forward_surround_avg(self):
        # test if surround works as expected on a toy example
        device = torch.device("cuda")
        I_cpu = np.array(
            [[[[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]],
              [[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]],
              [[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]]], dtype=np.float32)
        I = torch.from_numpy(I_cpu).to(device)
        I = torch.nn.functional.pad(I, (1, 1, 1, 1))
        w_s = torch.from_numpy(np.array(
            [[0, 0, 0], [1, 0, 1], [0, 0, 0]],
            dtype=np.float32)).to(device)
        w_c = torch.zeros_like(w_s).to(device)
        w_b = torch.zeros(3, device=device)
        expected_cpu = np.array(
            [[[[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]],
              [[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]],
              [[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]]]], dtype=np.float32)
        expected = torch.from_numpy(expected_cpu).to(device)
        O = center_surround.forward(I, w_c, w_s, w_b)[0]
        difference = torch.sum(expected - O)
        self.assertAlmostEqual(difference.item(), 0.0,
                               msg="Surround influence is wrong in forward pass.")

    def test_backward_simple(self):
        device = torch.device("cuda")
        batch_size = 1
        size = 3
        # Create some random input parameters
        I = torch.randn(batch_size, 1, size, size, requires_grad=True,
                        device=device, dtype=torch.double)
        w_c = torch.randn(1, 1, device=device, requires_grad=True,
                          dtype=torch.double)
        w_s = torch.randn(1, 1, device=device, requires_grad=True,
                          dtype=torch.double)
        w_b = torch.randn(1, device=device, requires_grad=True,
                          dtype=torch.double)
        O = csc.center_surround_convolution.apply(I, w_c, w_s, w_b)
        O.backward(torch.ones_like(O))

        dO_dI_gt = torch.zeros(size, size, device=device)
        dO_dI_gt[:, :] = w_s[0, 0]
        dO_dI_gt[1, 1] = w_c[0, 0]

        self.assertAlmostEqual((dO_dI_gt - I.grad).sum().item(), 0.0,
                               msg="dL_dI is not correct.", delta=1e-6)
        self.assertAlmostEqual((I[0, 0, 1, 1] - w_c.grad).item(), 0.0,
                               msg="dL_dw_c is not correct.",
                               delta=1e-6)
        w_s_gt = (torch.sum(I[0, 0, 0, :]) + torch.sum(I[0, 0, 2, :])
                  + I[0, 0, 1, 0] + I[0, 0, 1, 2])
        self.assertAlmostEqual((w_s_gt - w_s.grad).item(), 0.0,
                               msg="dL_dw_s is not correct.",
                               delta=1e-6)

    def test_backward_bias(self):
        device = torch.device("cuda")
        batch_size = 10
        size = 12
        # Create some random input parameters
        I = torch.randn(batch_size, 3, size, size, requires_grad=True,
                        device=device, dtype=torch.double)
        w_c = torch.randn(3, 3, device=device, requires_grad=True,
                          dtype=torch.double)
        w_s = torch.randn(3, 3, device=device, requires_grad=True,
                          dtype=torch.double)
        w_b = torch.randn(3, device=device, requires_grad=True,
                          dtype=torch.double)
        O = csc.center_surround_convolution.apply(I, w_c, w_s, w_b)
        O.backward(torch.ones_like(O))

        wb_grad = torch.ones_like(O).sum(dim=[0, 2, 3])
        diff = (w_b.grad - wb_grad).sum().item()

        self.assertAlmostEqual(diff, 0.0, delta=1e-6,
                               msg="dL_dw_b is not correct.")

    def test_backward_vs_torch(self):
        # run test against a csc version implemented with pytorch functions
        device = torch.device("cuda")
        for _ in range(10):
            batch_size = np.random.randint(1, 20)
            size = np.random.randint(3, 64)
            C_i = np.random.randint(1, 32)
            C_o = np.random.randint(1, 32)

            # Create some random input parameters
            I = torch.randn(batch_size, C_i, size, size, requires_grad=True,
                            device=device, dtype=torch.double)
            w_c = torch.randn(C_i, C_o, device=device, requires_grad=True,
                                dtype=torch.double)
            w_s = torch.randn(C_i, C_o, device=device, requires_grad=True,
                                dtype=torch.double)
            w_b = torch.randn(C_o, device=device, requires_grad=True,
                                dtype=torch.double)

            O_ours = csc.center_surround_convolution.apply(I, w_c, w_s, w_b)
            O_torch = pytorch_center_surround(I, w_c, w_s, w_b)
            forward_equal = torch.allclose(O_ours, O_torch)

            self.assertTrue(forward_equal, msg="Forward pass is wrong!")

            O_ours.backward(torch.ones_like(O_ours))
            our_dO_dw_c = w_c.grad.clone()
            our_dO_dw_s = w_s.grad.clone()
            our_dO_dw_b = w_b.grad.clone()
            our_dO_dI = I.grad.clone()

            w_c.grad.zero_()
            w_s.grad.zero_()
            w_b.grad.zero_()
            I.grad.zero_()
            O_torch.backward(torch.ones_like(O_torch))
            torch_dO_dw_c = w_c.grad.clone()
            torch_dO_dw_s = w_s.grad.clone()
            torch_dO_dw_b = w_b.grad.clone()
            torch_dO_dI = I.grad.clone()

            self.assertTrue(torch.allclose(our_dO_dw_b, torch_dO_dw_b),
                            "Gradient w.r.t. w_b is wrong.")
            self.assertTrue(torch.allclose(our_dO_dw_c, torch_dO_dw_c),
                            "Gradient w.r.t. w_c is wrong.")
            self.assertTrue(torch.allclose(our_dO_dw_s, torch_dO_dw_s),
                            "Gradient w.r.t. w_s is wrong.")
            self.assertTrue(torch.allclose(our_dO_dI, torch_dO_dI),
                            "Gradient w.r.t. I is wrong.")


if __name__ == '__main__':
    unittest.main()
