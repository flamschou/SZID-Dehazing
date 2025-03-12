from collections import namedtuple
import os
from typing import Any, Dict, List, Optional
from cv2.ximgproc import guidedFilter
from loguru import logger
from netZID.losses import StdLoss
from netZID.skip_model import skip
from utilsSZID.image_io import (
    np_to_torch,
    prepare_gt_img,
    prepare_hazy_image,
    save_image,
    torch_to_np,
)
from utilsZID.imresize import np_imresize
from netSZID.losses import AngularLossLayer
from netSZID.regression_model import FirstLayers, FinalLayer
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
import numpy as np
from tqdm import tqdm


import argparse

DehazeResult_psnr = namedtuple("DehazeResult", ["learned", "t", "a", "psnr"])
DehazeResult_ssim = namedtuple("DehazeResult", ["learned", "t", "a", "ssim"])


class Dehaze(object):
    def __init__(
        self,
        list_image_name: List[str],
        list_image: List[np.ndarray],
        list_gt_img: Optional[List[np.ndarray]] = None,
        list_val_img: Optional[List[np.ndarray]] = None,
        list_val_gt_img: Optional[List[np.ndarray]] = None,
        num_iter: int = 200,
        clip: bool = True,
        use_gpu: Optional[bool] = None,
        learning_rate: float = 0.001,
        pre_training: bool = False,
    ):
        self.list_image_name = list_image_name
        self.list_image = list_image
        if list_gt_img is None:
            self.list_gt_img = [np.array([]) for _ in list_image]
        else:
            self.list_gt_img = list_gt_img
        self.list_val_img = list_val_img
        if list_val_gt_img is None:
            self.list_val_gt_img = None
        else:
            self.list_val_gt_img = list_val_gt_img
        self.num_iter = num_iter
        self.ambient_val = None
        self.learning_rate = learning_rate
        self.current_result = None
        self.pre_training = pre_training

        self.storage_train_results = []
        self.storage_val_results = []

        # Determine device to use
        if use_gpu is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(
                "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            )

        self.clip = clip
        self.image_out = None
        self.mask_out = None
        self.ambient_out = None
        self.input_depth = 3

        self.angular_loss = AngularLossLayer().to(self.device)

        self.post = None
        self._init_all()

    def _init_images(self):
        self.array_originals_image = self.list_image.copy()
        self.list_image_torch = [
            np_to_torch(image).to(self.device) for image in self.list_image
        ]

        if self.list_val_img is not None:
            self.array_originals_val_image = self.list_val_img.copy()
            self.list_val_image_torch = [
                np_to_torch(image).to(self.device) for image in self.list_val_img
            ]

    def _init_nets(self):
        input_depth = self.input_depth
        pad = "reflection"

        image_net = skip(
            input_depth,
            3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode="bilinear",
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            act_fun="LeakyReLU",
        )
        self.image_net = image_net.to(self.device)

        mask_net = skip(
            input_depth,
            1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode="bilinear",
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            act_fun="LeakyReLU",
        )

        self.mask_net = mask_net.to(self.device)

    def _init_ambient(self):
        ambient_net = FirstLayers()
        self.ambient_net = ambient_net.to(self.device)
        self.list_final_layer = [
            FinalLayer(image.shape).to(self.device) for image in self.list_image
        ]
        if self.list_val_img is not None:
            self.list_val_final_layer = [
                FinalLayer(image.shape).to(self.device) for image in self.list_val_img
            ]

    def _init_parameters(self):
        parameters = [p for p in self.image_net.parameters()] + [
            p for p in self.mask_net.parameters()
        ]
        parameters += [p for p in self.ambient_net.parameters()]

        for final_layer in self.list_final_layer:
            for param in final_layer.parameters():
                if self.pre_training:
                    param.requires_grad = False
                parameters.append(param)

        self.parameters = parameters

    def _init_loss(self):
        self.mse_loss = torch.nn.MSELoss().to(self.device)
        self.blur_loss = StdLoss().to(self.device)

    def _init_inputs(self):
        self.list_image_net_inputs = [
            np_to_torch(image).to(self.device) for image in self.list_image
        ]
        self.list_mask_net_inputs = [
            np_to_torch(image).to(self.device) for image in self.list_image
        ]
        self.list_ambient_net_input = [
            np_to_torch(image).to(self.device) for image in self.list_image
        ]

        if self.list_val_img is not None:
            self.list_val_image_net_inputs = [
                np_to_torch(image).to(self.device) for image in self.list_val_img
            ]
            self.list_val_mask_net_inputs = [
                np_to_torch(image).to(self.device) for image in self.list_val_img
            ]
            self.list_val_ambient_net_input = [
                np_to_torch(image).to(self.device) for image in self.list_val_img
            ]

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def optimize(self):
        if self.device.type == "cuda":
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in tqdm(range(self.num_iter)):
            optimizer.zero_grad()
            self._optimization_closure()
            self._obtain_current_result(j)
            self._plot_closure(j)
            if self.list_val_img is not None:
                self._evaluate()
            optimizer.step()

    def _optimization_closure(self):
        # Forward passes
        self.list_image_out = [
            self.image_net(image_net_inputs)
            for image_net_inputs in self.list_image_net_inputs
        ]
        self.list_ambient_out = [
            final_layer(self.ambient_net(ambient_net_input))
            for final_layer, ambient_net_input in zip(
                self.list_final_layer, self.list_ambient_net_input
            )
        ]
        self.list_mask_out = [
            self.mask_net(mask_net_inputs)
            for mask_net_inputs in self.list_mask_net_inputs
        ]

        # Loss calculations
        lambda1, lambda2, lambda3, lambda4 = 200, 80000, 240, 1

        # Calculate blur loss - use torch.stack instead of torch.tensor
        blur_losses = [self.blur_loss(mask_out) for mask_out in self.list_mask_out]
        self.list_blur_out = torch.stack(blur_losses)

        # Calculate MSE losses
        mse_losses = []
        for mask_out, image_out, ambient_out, image_torch in zip(
            self.list_mask_out,
            self.list_image_out,
            self.list_ambient_out,
            self.list_image_torch,
        ):
            mask_out = mask_out.expand(-1, 3, -1, -1)
            ambient_out = ambient_out.view(1, 3, 1, 1).expand(
                -1, -1, mask_out.shape[2], mask_out.shape[3]
            )
            mse_losses.append(
                self.mse_loss(
                    mask_out * image_out + (1 - mask_out) * ambient_out, image_torch
                )
            )

        # Use torch.stack instead of torch.tensor
        self.list_mse_loss = torch.stack(mse_losses)

        # Calculate total loss components
        self.list_total_loss = lambda1 * self.list_mse_loss
        self.list_total_loss += lambda2 * self.list_blur_out

        # Calculate DCP priors and losses
        list_dcp_prior = [
            torch.min(image_out.permute(0, 2, 3, 1), dim=3)[0]
            for image_out in self.list_image_out
        ]

        dcp_losses = [
            self.mse_loss(dcp_prior, torch.zeros_like(dcp_prior))
            for dcp_prior in list_dcp_prior
        ]
        self.list_dcp_loss = torch.stack(dcp_losses)
        self.list_total_loss += lambda4 * self.list_dcp_loss

        # Calculate angular losses
        angular_losses = [
            self.angular_loss(image_out, image_torch)
            for image_out, image_torch in zip(
                self.list_image_out, self.list_image_torch
            )
        ]
        self.angular_loss_value = torch.stack(angular_losses)
        self.list_total_loss += lambda3 * self.angular_loss_value

        # Sum all losses
        self.total_loss = self.list_total_loss.sum()

        self.mean_total_loss = self.list_total_loss.mean()

        # Perform backpropagation on the total loss
        self.total_loss.backward()

    def _obtain_current_result(self, step) -> None:
        list_image_out_np = [
            np.clip(torch_to_np(image_out), 0, 1) for image_out in self.list_image_out
        ]
        list_mask_out_np = [
            np.clip(torch_to_np(mask_out), 0, 1) for mask_out in self.list_mask_out
        ]
        list_ambient_out_np = [
            np.clip(torch_to_np(ambient_out), 0, 1)
            for ambient_out in self.list_ambient_out
        ]
        list_mask_out_np = self.t_matting(list_mask_out_np)

        posts = []

        for image, mask_out_np, ambient_out_np in zip(
            list_image_out_np, list_mask_out_np, list_ambient_out_np
        ):
            ambient_out_np = ambient_out_np.reshape(3, 1, 1)
            post = np.clip(
                (image - ((1 - mask_out_np) * ambient_out_np)) / mask_out_np, 0, 1
            )
            posts.append(post)

        list_psnr = []
        list_ssim = []
        for i, gt_img in enumerate(self.list_gt_img):
            if gt_img.size != 0:
                psnr = compare_psnr(gt_img, posts[i])
                ssims = compare_ssim(
                    gt_img,
                    posts[i],
                    data_range=gt_img.max() - gt_img.min(),
                    channel_axis=0,
                )
            else:
                psnr = None
                ssims = None
            list_psnr.append(psnr)
            list_ssim.append(ssims)

        self.list_current_result = [
            DehazeResult_psnr(
                learned=image_out_np,
                t=mask_out_np,
                a=ambient_out_np.reshape(3, 1, 1),
                psnr=psnr,
            )
            for image_out_np, mask_out_np, ambient_out_np, psnr in zip(
                list_image_out_np, list_mask_out_np, list_ambient_out_np, list_psnr
            )
        ]
        self.list_temp = [
            DehazeResult_ssim(
                learned=image_out_np, t=mask_out_np, a=ambient_out_np, ssim=ssims
            )
            for image_out_np, mask_out_np, ambient_out_np, ssims in zip(
                list_image_out_np, list_mask_out_np, list_ambient_out_np, list_ssim
            )
        ]

    def _plot_closure(self, step):
        """
        Log metrics to Wandb instead of console

        :param step: the number of the iteration
        :return:
        """
        # Prepare metrics for logging
        metrics = {
            "total_loss": self.mean_total_loss.item(),
            "dcp_loss": self.list_dcp_loss.mean().item(),
            "angular_loss": self.angular_loss_value.mean().item(),
            "mse_loss": self.list_mse_loss.mean().item(),
            "blur_loss": self.list_blur_out.mean().item(),
        }

        # Log per-image metrics
        for i, (loss, blur_out) in enumerate(
            zip(self.list_total_loss, self.list_blur_out)
        ):
            metrics[f"image_{i}/loss"] = loss.item()
            metrics[f"image_{i}/blur_loss"] = blur_out.item()

        # Log PSNR and SSIM if ground truth is available
        ground_truth_available = True
        for i, (gt_img, temp, current_result) in enumerate(
            zip(self.list_gt_img, self.list_temp, self.list_current_result)
        ):
            if gt_img.size != 0:
                metrics[f"image_{i}/ssim"] = temp.ssim
                metrics[f"image_{i}/psnr"] = current_result.psnr
            else:
                ground_truth_available = False

        # Log mean PSNR and SSIM if all images have ground truth
        if ground_truth_available:
            sum_ssim = np.mean([temp.ssim for temp in self.list_temp])
            sum_psnr = np.mean(
                [current_result.psnr for current_result in self.list_current_result]
            )
            metrics["ssim"] = float(sum_ssim)
            metrics["psnr"] = float(sum_psnr)

        self.storage_train_results.append(metrics)

    def _evaluate(self):
        """
        Performs forward pass only to compute validation loss without backpropagation.
        """
        self.image_net.eval()
        self.ambient_net.eval()
        self.mask_net.eval()

        total_losses = []
        mse_losses = []
        blur_losses = []
        dcp_losses = []
        angular_losses = []

        if self.list_val_img is None:
            logger.warning(
                "No validation images provided. Using training images as validation images."
            )
            list_val_image_net_inputs = self.list_image_net_inputs
        else:
            list_val_image_net_inputs = self.list_val_image_net_inputs

        with torch.no_grad():
            # Forward passes
            list_val_image_out = [
                self.image_net(val_image_net_inputs)
                for val_image_net_inputs in list_val_image_net_inputs
            ]
            list_val_ambient_out = [
                val_final_layer(self.ambient_net(val_ambient_net_input))
                for val_final_layer, val_ambient_net_input in zip(
                    self.list_val_final_layer, self.list_val_ambient_net_input
                )
            ]
            list_val_mask_out = [
                self.mask_net(mask_val_net_inputs)
                for mask_val_net_inputs in self.list_val_mask_net_inputs
            ]

            # Loss calculations
            lambda1, lambda2, lambda3, lambda4 = 200, 80000, 240, 1

            # Calculate blur loss
            batch_blur_losses = [
                self.blur_loss(mask_out) for mask_out in list_val_mask_out
            ]
            batch_blur_out = torch.stack(batch_blur_losses)
            blur_losses.append(batch_blur_out.mean().item())

            # Calculate MSE losses
            batch_mse_losses = []
            for mask_out, image_out, ambient_out, val_image_torch in zip(
                list_val_mask_out,
                list_val_image_out,
                list_val_ambient_out,
                self.list_val_image_torch,
            ):
                mask_out = mask_out.expand(-1, 3, -1, -1)
                ambient_out = ambient_out.view(1, 3, 1, 1).expand(
                    -1, -1, mask_out.shape[2], mask_out.shape[3]
                )

                batch_mse_losses.append(
                    self.mse_loss(
                        mask_out * image_out + (1 - mask_out) * ambient_out,
                        val_image_torch,
                    )
                )

            batch_mse_loss = torch.stack(batch_mse_losses)
            mse_losses.append(batch_mse_loss.mean().item())

            # Calculate total loss components
            batch_total_loss = lambda1 * batch_mse_loss
            batch_total_loss += lambda2 * batch_blur_out

            # Calculate DCP priors and losses
            list_dcp_prior = [
                torch.min(image_out.permute(0, 2, 3, 1), dim=3)[0]
                for image_out in list_val_image_out
            ]
            batch_dcp_losses = [
                self.mse_loss(dcp_prior, torch.zeros_like(dcp_prior))
                for dcp_prior in list_dcp_prior
            ]
            batch_dcp_loss = torch.stack(batch_dcp_losses)
            dcp_losses.append(batch_dcp_loss.mean().item())
            batch_total_loss += lambda4 * batch_dcp_loss

            # Calculate angular losses
            batch_angular_losses = [
                self.angular_loss(val_image_out, val_image_torch)
                for val_image_out, val_image_torch in zip(
                    list_val_image_out, self.list_val_image_torch
                )
            ]
            batch_angular_loss_value = torch.stack(batch_angular_losses)
            angular_losses.append(batch_angular_loss_value.mean().item())
            batch_total_loss += lambda3 * batch_angular_loss_value

            # Sum batch total loss
            total_losses.append(batch_total_loss.mean().item())

        self.storage_val_results.append(
            {
                "total_loss": total_losses,
                "mse_loss": np.mean(mse_losses),
                "blur_loss": np.mean(blur_losses),
                "dcp_loss": np.mean(dcp_losses),
                "angular_loss": np.mean(angular_losses),
            }
        )

        list_val_image_out_np = [
            np.clip(torch_to_np(val_image_out), 0, 1)
            for val_image_out in list_val_image_out
        ]
        list_val_mask_out_np = [
            np.clip(torch_to_np(val_mask_out), 0, 1)
            for val_mask_out in list_val_mask_out
        ]
        list_val_ambient_out_np = [
            np.clip(torch_to_np(val_ambient_out), 0, 1)
            for val_ambient_out in list_val_ambient_out
        ]
        list_val_mask_out_np = self.t_matting(list_val_mask_out_np, validation=True)

        posts = []

        for val_image, val_mask_out_np, val_ambient_out_np in zip(
            list_val_image_out_np, list_val_mask_out_np, list_val_ambient_out_np
        ):
            val_ambient_out_np = val_ambient_out_np.reshape(3, 1, 1)
            post = np.clip(
                (val_image - ((1 - val_mask_out_np) * val_ambient_out_np))
                / val_mask_out_np,
                0,
                1,
            )
            posts.append(post)

        list_psnr = []
        list_ssim = []

        if self.list_val_gt_img is None:
            logger.warning(
                "No validation ground truth images provided. Skipping PSNR and SSIM calculation."
            )
            return

        for i, val_gt_img in enumerate(self.list_val_gt_img):
            if val_gt_img.size != 0:
                psnr = compare_psnr(val_gt_img, posts[i])
                ssims = compare_ssim(
                    val_gt_img,
                    posts[i],
                    data_range=val_gt_img.max() - val_gt_img.min(),
                    channel_axis=0,
                )
            else:
                psnr = None
                ssims = None
            list_psnr.append(psnr)
            list_ssim.append(ssims)

        self.list_val_current_result = [
            DehazeResult_psnr(
                learned=val_image_out_np,
                t=val_mask_out_np,
                a=val_ambient_out_np.reshape(3, 1, 1),
                psnr=psnr,
            )
            for val_image_out_np, val_mask_out_np, val_ambient_out_np, psnr in zip(
                list_val_image_out_np,
                list_val_mask_out_np,
                list_val_ambient_out_np,
                list_psnr,
            )
        ]
        self.list_val_temp = [
            DehazeResult_ssim(
                learned=val_image_out_np,
                t=val_mask_out_np,
                a=val_ambient_out_np,
                ssim=ssims,
            )
            for val_image_out_np, val_mask_out_np, val_ambient_out_np, ssims in zip(
                list_val_image_out_np,
                list_val_mask_out_np,
                list_val_ambient_out_np,
                list_ssim,
            )
        ]

        # store the mean pnsr and ssim in the storage_val_results

        self.storage_val_results[-1]["psnr"] = np.mean(list_psnr)
        self.storage_val_results[-1]["ssim"] = np.mean(list_ssim)

    def finalize(self):
        self.list_final_t_map = [
            np_imresize(current_result.t, output_shape=original_image.shape[1:])
            for current_result, original_image in zip(
                self.list_current_result, self.array_originals_image
            )
        ]
        logger.debug(f"number original images: {len(self.array_originals_image)}")
        self.list_final_a = [
            np_imresize(current_result.a, output_shape=original_image.shape[1:])
            for current_result, original_image in zip(
                self.list_current_result, self.array_originals_image
            )
        ]
        list_mask_out_np = self.t_matting(self.list_final_t_map)
        list_post = [
            np.clip(
                (original_image - ((1 - mask_out_np) * final_a)) / mask_out_np,
                0,
                1,
            )
            for original_image, mask_out_np, final_a in zip(
                self.array_originals_image, list_mask_out_np, self.list_final_a
            )
        ]

        for image_name, post in zip(self.list_image_name, list_post):
            save_image(f"results/{image_name}_dehazed", post)

    def t_matting(self, list_mask_out_np, validation: bool = False):
        if not validation:
            list_refine_t = [
                guidedFilter(
                    original_image.transpose(1, 2, 0).astype(np.float32),
                    mask_out_np[0].astype(np.float32),
                    50,
                    1e-4,
                )
                for original_image, mask_out_np in zip(
                    self.array_originals_image, list_mask_out_np
                )
            ]
        else:
            list_refine_t = [
                guidedFilter(
                    original_image.transpose(1, 2, 0).astype(np.float32),
                    mask_out_np[0].astype(np.float32),
                    50,
                    1e-4,
                )
                for original_image, mask_out_np in zip(
                    self.array_originals_val_image, list_mask_out_np
                )
            ]
        if self.clip:
            return [np.array([np.clip(refine_t, 0.1, 1)]) for refine_t in list_refine_t]
        else:
            return [np.array([np.clip(refine_t, 0, 1)]) for refine_t in list_refine_t]

    def save_weights(
        self,
        path: Optional[str] = None,
        epoch: Optional[int] = None,
        additional_info: Optional[str] = None,
    ):
        """
        Save model weights to disk.

        Args:
            path (str): Base path to save the weights.
            epoch (int, optional): Current epoch number to include in filename.
            additional_info (dict, optional): Additional information to save with weights.
        """
        # Create folder if it doesn't exist
        import os

        if path is None:
            path = "weights/weights"

        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

        # Construct filename
        filename = path
        if epoch is not None:
            filename = f"{path}_epoch_{epoch}"

        # Prepare state dictionary with all networks

        state_dict: Dict[str, Any] = {
            "image_net": self.image_net.state_dict(),
            "mask_net": self.mask_net.state_dict(),
            "ambient_net": self.ambient_net.state_dict(),
        }

        # Add optimization info if available
        state_dict["learning_rate"] = self.learning_rate

        # Add additional info
        if additional_info:
            state_dict["additional_info"] = additional_info

        # Save to disk
        torch.save(state_dict, f"{filename}.pth")
        logger.info(f"Model weights saved to {filename}.pth")

    def load_weights(
        self, path: str = "weights/weights.pth", strict=True, eval_mode=False
    ):
        """
        Load model weights from disk.

        Args:
            path (str): Path to the saved weights file.
            strict (bool): Whether to strictly enforce that the keys in state_dict match
                          the keys returned by this module's state_dict() function.

        Returns:
            dict: Additional info that was stored with the weights, if any.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found at {path}")

        # Load state dict
        checkpoint = torch.load(path, map_location=self.device)

        # Load network weights
        if "image_net" in checkpoint and self.image_net is not None:
            self.image_net.load_state_dict(checkpoint["image_net"], strict=strict)
        if "mask_net" in checkpoint and self.mask_net is not None:
            self.mask_net.load_state_dict(checkpoint["mask_net"], strict=strict)
        if "ambient_net" in checkpoint and self.ambient_net is not None:
            self.ambient_net.load_state_dict(checkpoint["ambient_net"], strict=strict)

        # Update learning rate if saved
        if "learning_rate" in checkpoint:
            self.learning_rate = checkpoint["learning_rate"]

        # Re-initialize parameters to include loaded models
        self._init_parameters()

        if not eval_mode:
            logger.info(f"Model weights loaded from {path}")

        # Return any additional info that was stored
        return checkpoint.get("additional_info", None)


def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


# def dehaze(image, gt_img, nameout, num_iter=200, use_gpu=None):
#     dh = Dehaze(nameout, image, gt_img, num_iter, clip=True, use_gpu=use_gpu)

#     dh.optimize()
#     dh.finalize()

#     save_image(nameout + "_original", np.clip(image, 0, 1))

#     if gt_img.size != 0:
#         psnr = dh.current_result.psnr
#         ssim = dh.temp.ssim
#         print(psnr, ssim)

#     params_image_net = count_trainable_parameters(dh.image_net)
#     params_mask_net = count_trainable_parameters(dh.mask_net)
#     params_ambient_net = count_trainable_parameters(dh.ambient_net)

#     total_params = params_image_net + params_mask_net + params_ambient_net

#     print("J-Net parameters: %d" % params_image_net)
#     print("T-Net parameters: %d" % params_mask_net)
#     print("A-Net parameters: %d" % params_ambient_net)
#     print("SZID parameters: %d" % total_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SZID Dehazing Script")
    parser.add_argument(
        "--num_iter", type=int, default=200, help="Number of iterations"
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=False, help="Use GPU (True/False)"
    )

    args = parser.parse_args()

    num_iter = args.num_iter
    use_gpu = args.use_gpu
    train = os.listdir("train_set/hazy/")

    list_hazy_img = []
    list_gt_img = []
    for image in train:
        hazy_img = prepare_hazy_image("train_set/hazy/" + image)
        gt_img = prepare_gt_img("train_set/GT/" + image[:-9] + "_GT.jpg", SOTS=False)
        list_hazy_img.append(hazy_img)
        list_gt_img.append(gt_img)

    val = os.listdir("val_set/hazy/")

    list_hazy_val_img = []
    list_gt_val_img = []
    for image in val:
        hazy_img = prepare_hazy_image("val_set/hazy/" + image)
        gt_img = prepare_gt_img("val_set/GT/" + image[:-9] + "_GT.jpg", SOTS=False)
        list_hazy_val_img.append(hazy_img)
        list_gt_val_img.append(gt_img)

    dh = Dehaze(
        [f"output_{i}" for i in range(len(list_hazy_img))],
        list_hazy_img,
        list_gt_img,
        list_hazy_val_img,
        list_gt_val_img,
        num_iter=num_iter,
        clip=True,
        use_gpu=use_gpu,
        pre_training=True,
    )
    dh.optimize()
    dh.finalize()
    dh.save_weights()

    logger.info("Dehazing complete")
