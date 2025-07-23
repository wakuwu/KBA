import torch
import torch.nn.functional as F


class LossTools(object):
    """Utility class containing various loss computation tools."""

    def __init__(self):
        super().__init__()

    @staticmethod
    @torch.no_grad()
    def get_bounding_box_center_and_size(mask):
        """
        Calculate the bounding box center, size and coordinates from a mask.
        
        Args:
            mask: Binary mask tensor of shape (H, W)
            
        Returns:
            tuple: (center_coords, width_height, min_max_coords) or None if mask is empty
                - center_coords: Center coordinates of the bounding box
                - width_height: Width and height of the bounding box
                - min_max_coords: Stack of min and max coordinates
        """
        # Find all non-zero indices where mask equals 1
        non_zero_indices = torch.nonzero(mask.squeeze(), as_tuple=False)
        # Return None if no mask elements equal 1 are found
        if non_zero_indices.numel() == 0:
            return None
        # Get minimum and maximum coordinates
        min_coords = torch.min(non_zero_indices, dim=0)[0].flip(0)
        max_coords = torch.max(non_zero_indices, dim=0)[0].flip(0)
        # Calculate center coordinates of the bounding box
        center_coords = (min_coords + max_coords) / 2
        # Calculate width and height of the bounding box
        width_height = max_coords - min_coords + 1
        return center_coords, width_height, torch.stack([min_coords, max_coords], dim=0)

    @staticmethod
    @torch.no_grad()
    def calculate_masked_min_max(input, mask):
        """
        Calculate the minimum and maximum values in a tensor based on a mask.
        
        Args:
            input: Input tensor to find min/max values
            mask: Boolean mask indicating valid regions
            
        Returns:
            tuple: (input_max_values, input_min_values) - tensors with max and min values
        """
        input_max = input.masked_fill(~mask, float('-inf'))
        input_min = input.masked_fill(~mask, float('inf'))
        input_max_values = input_max.flatten(start_dim=-2).max(-1).values[..., None, None].expand_as(input)
        input_min_values = input_min.flatten(start_dim=-2).min(-1).values[..., None, None].expand_as(input)
        return input_max_values, input_min_values

    @staticmethod
    @torch.no_grad()
    def split_masks(masks, center_coords, thetas):
        """
        Split masks into regions based on directional angles.
        
        Args:
            masks: Input masks tensor
            center_coords: Center coordinates for each mask
            thetas: Angles in degrees [0, 90] for different angle groups
                   Format: [[view_1, view_2, ..., view_n], [group2]]
                   where n is the number of angle groups
                   
        Returns:
            torch.Tensor: Shape [b, c, h, w, 2(xy positive direction mask), 3(thetas)]
        """
        device = masks.device
        b, _, h, w = masks.shape
        n = thetas.shape[1]
        center_coords = center_coords.permute([1, 0])
        # Create a coordinate grid
        y_grid, x_grid = torch.meshgrid(torch.arange(h, device=device),
                        torch.arange(w, device=device), indexing='ij')
        y_grid_b = y_grid[..., None, None].repeat([1, 1, b, n])
        x_grid_b = x_grid[..., None, None].repeat([1, 1, b, n])
        # Calculate the mask for the first line (x-axis as the divider)
        tan_thetas = torch.tan(torch.deg2rad(thetas)).to(device)
        line1_mask = (y_grid_b - center_coords[1][..., None].repeat([1, n])) - tan_thetas * (
            x_grid_b - center_coords[0][..., None].repeat([1, n]))
        # Two masks for the regions divided by the first line
        mask1_part = (line1_mask >= 0)[None, ..., None, :].permute([3, 0, 1, 2, 4, 5])
        # Calculate the mask for the line perpendicular to the first line
        line2_mask = (y_grid_b - center_coords[1][..., None].repeat([1, n])) + 1 / tan_thetas * (
            x_grid_b - center_coords[0][..., None].repeat([1, n]))
        # Two masks for the regions divided by the second line (y-axis as the divider)
        mask2_part = (line2_mask >= 0)[None, ..., None, :].permute([3, 0, 1, 2, 4, 5])
        return torch.cat([mask2_part, mask1_part], dim=-2)

    @staticmethod
    @torch.no_grad()
    def axis_direction_vectors(thetas):
        """
        Calculate axis direction vectors from given angles.
        
        Args:
            thetas: Angle values in degrees
            
        Returns:
            torch.Tensor: Stack of x and y axis direction vectors
        """
        thetas_rad = torch.deg2rad(thetas)
        x_axis_vector = torch.stack([torch.cos(thetas_rad), torch.sin(thetas_rad)], dim=0)
        y_axis_vector = torch.stack([torch.cos(thetas_rad + torch.pi / 2), torch.sin(thetas_rad + torch.pi / 2)], dim=0)
        return torch.stack([x_axis_vector, y_axis_vector], dim=0)

    @staticmethod
    @torch.no_grad()
    def vis_vector_direction(images, center_coords, xyz_vector):
        """
        Visualize vector directions on images for debugging purposes.
        
        Args:
            images: Input images tensor
            center_coords: Center coordinates for visualization
            xyz_vector: XYZ direction vectors to visualize
            
        Returns:
            None (displays visualizations)
        """
        for b_idx in range(len(images)):
            images_b = images[b_idx]
            xyz_vector_b = xyz_vector[b_idx]
            for view_idx in range(len(images_b)):
                images_b_v = images_b[view_idx]
                xyz_vector_b_v = xyz_vector_b[view_idx]
                center_coords_v = center_coords[view_idx]
                for axis_idx in range(3):
                    images_b_v_a = images_b_v[axis_idx]
                    xyz_vector_b_v_a = xyz_vector_b_v[axis_idx]
                    from utils import heatmap_direction
                    heatmap_direction(images_b_v_a, is_show=True, start_point=center_coords_v.cpu().numpy(),
                                      direction_vector=xyz_vector_b_v_a.cpu().numpy(), length=80, arrow_width=15)
        return


class LossAttacks(object):
    """Class containing attack-related loss functions."""
    
    @staticmethod
    def loss_chaos_direction(batch, outputs, **kwargs):
        """
        Calculate chaos direction loss for adversarial attacks.
        
        This function computes a loss that encourages chaotic directional patterns
        in the carrier regions by analyzing directional consistency across different
        angular splits of the mask regions.
        
        Args:
            batch: Dictionary containing batch data including masks and axis directions
            outputs: Model outputs tensor stack
            **kwargs: Additional keyword arguments
            
        Returns:
            dict: Dictionary containing the computed loss value
        """
        # Pre-processing
        outputs = torch.stack(outputs, dim=0)
        # Normalize carrier coordinates for convenient axis direction calculation
        outputs_max, outputs_min = LossTools.calculate_masked_min_max(outputs[:, :, :3],
                                                                      batch['masks_carrier_wo_victim'].bool())
        xyz_coords = (outputs[:, :, :3] - outputs_min) / (outputs_max - outputs_min)
        # Calculate split masks
        center_coords = []
        for mask_view in batch['masks_carrier_wo_victim']:
            center_coord, _, _ = LossTools.get_bounding_box_center_and_size(mask_view)
            center_coords.append(center_coord)
        center_coords = torch.stack(center_coords, dim=0)
        theta_list = [0., 30., 60.]
        thetas = torch.tensor([theta_list for _ in range(outputs.shape[1])], device=outputs.device)
        split_masks = LossTools.split_masks(batch['masks_carrier_wo_victim'], center_coords, thetas)
        axis_direction_vectors = LossTools.axis_direction_vectors(thetas[0])
        # Calculate direct similarity loss
        split_masks_p = \
            (split_masks * batch['masks_carrier_wo_victim'][..., None, None].repeat([1, 1, 1, 1, 2, len(theta_list)]))[
                None, ...].repeat([2, 1, 3, 1, 1, 1, 1])
        xyz_positive = (split_masks_p * xyz_coords[..., None, None].repeat(
            [1, 1, 1, 1, 1, 2, len(theta_list)])).flatten(
            start_dim=3, end_dim=4).sum(3) / split_masks_p.flatten(start_dim=3, end_dim=4).sum(3)
        split_masks_n = \
            ((~split_masks) * batch['masks_carrier_wo_victim'][..., None, None].repeat(
                [1, 1, 1, 1, 2, len(theta_list)]))[
                None, ...].repeat([2, 1, 3, 1, 1, 1, 1])
        xyz_negative = (split_masks_n * xyz_coords[..., None, None].repeat(
            [1, 1, 1, 1, 1, 2, len(theta_list)])).flatten(
            start_dim=3, end_dim=4).sum(3) / split_masks_n.flatten(start_dim=3, end_dim=4).sum(3)
        xyz_vector = xyz_positive - xyz_negative
        xyz_vector_true = (xyz_vector[..., None, :].repeat([1, 1, 1, 1, 2, 1]) * axis_direction_vectors).sum(-3)
        xyz_vector_true_correct = xyz_vector_true.mean(-1)
        # # Debug visualization (uncomment for debugging)
        # LossTools.vis_vector_direction(outputs, center_coords, xyz_vector_true_correct)
        loss_chaos_direction = F.cosine_similarity(xyz_vector_true_correct[[0, 1], [1, 0]], batch["axis_direction"][[0, 1], [0, 1]], dim=-1).mean()
        return dict(loss_chaos_direction=loss_chaos_direction)
        